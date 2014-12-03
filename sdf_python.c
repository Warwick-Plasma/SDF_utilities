#include <float.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <structmember.h>
#include "sdf.h"

#if PY_MAJOR_VERSION < 3
    #define PyInt_FromLong PyLong_FromLong
#endif

static const int typemap[] = {
    0,
    PyArray_UINT32,
    PyArray_UINT64,
    PyArray_FLOAT,
    PyArray_DOUBLE,
#ifdef NPY_FLOAT128
    PyArray_FLOAT128,
#else
    0,
#endif
    PyArray_CHAR,
    PyArray_CHAR,
};


typedef struct {
    PyObject_HEAD
    sdf_file_t *h;
} SDFObject;


static PyObject *
SDF_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    sdf_file_t *h;
    const char *file;
    SDFObject *self;
    int convert = 0, use_mmap = 1, mode = SDF_READ;
    comm_t comm = 0;

    if (!PyArg_ParseTuple(args, "s|i", &file, &convert))
        return NULL;

    self = (SDFObject*)type->tp_alloc(type, 0);
    if (self == NULL)
        return NULL;

    h = sdf_open(file, comm, mode, use_mmap);
    self->h = h;
    if (!self->h) {
        Py_DECREF(self);
        return NULL;
    }

    if (convert) h->use_float = 1;

    return (PyObject*)self;
}


static void
SDF_dealloc(PyObject* self)
{
    SDFObject *pyo = (SDFObject*)self;
    if (pyo->h) sdf_close(pyo->h);
    self->ob_type->tp_free(self);
}


static void setup_mesh(sdf_file_t *h, PyObject *dict)
{
    sdf_block_t *b = h->current_block;
    int i, n, ndims;
    size_t l1, l2;
    char *label;
    void *grid;
    PyObject *sub;
    npy_intp dims[3] = {0,0,0};

    sdf_read_data(h);

    for (n = 0; n < b->ndims; n++) {
        ndims = (int)b->dims[n];

        l1 = strlen(b->name);
        l2 = strlen(b->dim_labels[n]);
        label = malloc(l1 + l2 + 2);
        memcpy(label, b->name, l1);
        label[l1] = '/';
        memcpy(label+l1+1, b->dim_labels[n], l2+1);

        l1 = strlen(b->id);
        grid = b->grids[n];
        /* Hack to node-centre the cartesian grid */
        if (strncmp(b->id, "grid", l1+1) == 0) {
            ndims--;
            if (b->datatype_out == SDF_DATATYPE_REAL4) {
                float v1, v2, *ptr, *out;
                ptr = grid;
                grid = out = malloc(ndims * sizeof(*out));
                for (i = 0; i < ndims; i++) {
                    v1 = *ptr;
                    v2 = *(ptr+1);
                    *out++ = 0.5 * (v1 + v2);
                    ptr++;
                }
            } else {
                double v1, v2, *ptr, *out;
                ptr = grid;
                grid = out = malloc(ndims * sizeof(*out));
                for (i = 0; i < ndims; i++) {
                    v1 = *ptr;
                    v2 = *(ptr+1);
                    *out++ = 0.5 * (v1 + v2);
                    ptr++;
                }
            }

            dims[0] = ndims;
            sub = PyArray_NewFromDescr(&PyArray_Type,
                PyArray_DescrFromType(typemap[b->datatype_out]), 1,
                dims, NULL, grid, NPY_F_CONTIGUOUS, NULL);
            PyDict_SetItemString(dict, label, sub);
            Py_DECREF(sub);

            /* Now add the original grid with "_node" appended */
            ndims++;
            grid = b->grids[n];

            l1 = strlen(b->name);
            l2 = strlen(b->dim_labels[n]);
            label = malloc(l1 + l2 + 2 + 5);
            memcpy(label, b->name, l1);
            memcpy(label+l1, "_node", 5);
            label[l1+5] = '/';
            memcpy(label+l1+6, b->dim_labels[n], l2+1);
        }

        dims[0] = ndims;
        sub = PyArray_NewFromDescr(&PyArray_Type,
            PyArray_DescrFromType(typemap[b->datatype_out]), 1,
            dims, NULL, grid, NPY_F_CONTIGUOUS, NULL);
        PyDict_SetItemString(dict, label, sub);
        Py_DECREF(sub);
    }
}


static void setup_lagrangian_mesh(sdf_file_t *h, PyObject *dict)
{
    sdf_block_t *b = h->current_block;
    int n;
    size_t l1, l2;
    char *label;
    void *grid;
    PyObject *sub;
    npy_intp dims[3] = {0,0,0};

    sdf_read_data(h);

    for (n = 0; n < b->ndims; n++) dims[n] = (int)b->dims[n];

    for (n = 0; n < b->ndims; n++) {
        l1 = strlen(b->name);
        l2 = strlen(b->dim_labels[n]);
        label = malloc(l1 + l2 + 2);
        memcpy(label, b->name, l1);
        label[l1] = '/';
        memcpy(label+l1+1, b->dim_labels[n], l2+1);

        l1 = strlen(b->id);
        grid = b->grids[n];

        sub = PyArray_NewFromDescr(&PyArray_Type,
            PyArray_DescrFromType(typemap[b->datatype_out]), b->ndims,
            dims, NULL, grid, NPY_F_CONTIGUOUS, NULL);
        PyDict_SetItemString(dict, label, sub);
        Py_DECREF(sub);
    }
}



static void extract_station_time_histories(sdf_file_t *h, PyObject *kw,
      PyObject *dict)
{
   PyObject *stations = PyDict_GetItemString(kw, "stations"),
            *variables = PyDict_GetItemString(kw, "variables"),
            *py_t0 = PyDict_GetItemString(kw, "t0"),
            *py_t1 = PyDict_GetItemString(kw, "t1");
   Py_ssize_t nvars, i, nstat;
   PyObject *sub;
   char **var_names, *timehis, *v, *key;
   double t0, t1;
   long *stat;
   int *size, *offset, nrows, row_size, j;
   sdf_block_t *b;
   npy_intp dims[1];

   /* Stop gcc complaining about 'may be used uninitialised...' */
   stat = NULL;
   nstat = 0;

   if ( !stations ) {
      nstat = 1;
      stat = (long *)malloc(sizeof(long));
      stat[0] = 0;
   } else if ( PyInt_Check(stations) ) {
      nstat = 1;
      stat = (long *)malloc(sizeof(long));
      stat[0] = PyInt_AsLong(stations) - 1;
   } else if ( PySequence_Check(stations) ) {
      sub = PySequence_List(stations);
      PyList_Sort(sub);
      nstat = PySequence_Length(sub);
      stat = (long *)calloc(nstat, sizeof(long));
      j = 0;
      for ( i=0; i<nstat; i++ ) {
         stat[j] = PyInt_AsLong(PySequence_GetItem(sub, i)) - 1;
         if ( PyErr_Occurred() ) {
            PyErr_SetString(PyExc_TypeError,
                  "'stations' keyword must be an integer or list of integers");
            free(stat);
            return;
         }
         if ( i==0 || stat[j]!=stat[j-1] )
            j++;
      }
      nstat = j;
      Py_DECREF(sub);
   } else {
      PyErr_SetString(PyExc_TypeError,
            "'stations' keyword must be an integer or list of integers");
   }


   if ( !variables )
      return;

   nvars = PySequence_Length(variables);
   if ( nvars<0 ) {
      PyErr_SetString(PyExc_TypeError,
            "'variables' keyword must be a string or list of strings");
      free(stat);
      return;
   }
   if ( !nvars ) {
      free(stat);
      return;
   }

   if ( PyString_Check(variables) ) {
      nvars = 1;
      var_names = (char **)malloc(sizeof(char *));
      var_names[0] = PyString_AsString(variables);
   } else {
      var_names = (char **)malloc(nvars*sizeof(char *));
      for ( i=0; i<nvars; i++ ) {
         var_names[i] = PyString_AsString(PySequence_GetItem(variables, i));
         if ( !var_names[i] ) {
            free(var_names);
            free(stat);
            PyErr_SetString(PyExc_TypeError,
                  "'variables' keyword must be a string or list of strings");
            return;
         }
      }
   }

   if ( !py_t0 )
      t0 = -DBL_MAX;
   else {
      t0 = PyFloat_AsDouble(py_t0);
      if ( PyErr_Occurred() ) {
         PyErr_SetString(PyExc_TypeError, "'t0' keyword must be a number");
         free(var_names);
         free(stat);
         return;
      }
   }
   if ( !py_t1 )
      t1 = DBL_MAX;
   else {
      t1 = PyFloat_AsDouble(py_t1);
      if ( PyErr_Occurred() ) {
         PyErr_SetString(PyExc_TypeError, "'t1' keyword must be a number");
         free(var_names);
         free(stat);
         return;
      }
   }

   offset = (int *)calloc(nstat*nvars+1, sizeof(int));
   size = (int *)calloc(nstat*nvars+1, sizeof(int));
   if ( sdf_read_station_timehis(h, stat, nstat, var_names, nvars, t0, t1,
            &timehis, size, offset, &nrows, &row_size) ) {
      free(var_names);
      free(size);
      free(offset);
      free(stat);
      return;
   }

   b = h->current_block;
   key = malloc(2*h->string_length+1);
   dims[0] = nrows;

   /* Handle 'Time' as a special case */
   sub = PyArray_SimpleNewFromData(1, dims, typemap[b->variable_types[0]],
         timehis);

   sprintf(key, "%s/Time", b->name);

   PyDict_SetItemString(dict, key, sub);
   Py_DECREF(sub);

   v = timehis + nrows * size[0];
   for ( i=1; i<=nstat*nvars; i++ ) {
      if ( !size[i] )
         continue;

      sub = PyArray_SimpleNewFromData(
            1, dims, typemap[b->variable_types[i]], v);

      sprintf(key, "%s/Station_%ld/%s", b->name, 1+stat[(int)(i-1)/nvars],
            var_names[(i-1)%nvars]);

      printf("%s\n", key);

      PyDict_SetItemString(dict, key, sub);
      Py_DECREF(sub);

      v += nrows * size[i];
   }

   free(var_names);
   free(size);
   free(key);
   free(stat);
}


#define SET_ENTRY(type,value) do { \
        PyObject *sub; \
        sub = Py_BuildValue(#type, h->value); \
        PyDict_SetItemString(dict, #value, sub); \
        Py_DECREF(sub); \
    } while (0)

#define SET_BOOL(value) \
    if (h->value) PyDict_SetItemString(dict, #value, Py_True); \
    else PyDict_SetItemString(dict, #value, Py_False)

static PyObject *fill_header(sdf_file_t *h)
{
    PyObject *dict;

    dict = PyDict_New();

    SET_ENTRY(i, file_version);
    SET_ENTRY(i, file_revision);
    SET_ENTRY(s, code_name);
    SET_ENTRY(i, step);
    SET_ENTRY(d, time);
    SET_ENTRY(i, jobid1);
    SET_ENTRY(i, jobid2);
    SET_ENTRY(i, code_io_version);
    SET_BOOL(restart_flag);
    SET_BOOL(other_domains);

    return dict;
}

static PyObject* SDF_read(SDFObject *self, PyObject *args, PyObject *kw)
{
    sdf_file_t *h;
    sdf_block_t *b;
    PyObject *dict, *sub;
    int i, n;
    double dd;
    long il;
    long long ll;
    npy_intp dims[3] = {0,0,0};

    h = self->h;
    sdf_read_blocklist(h);
    dict = PyDict_New();

    // Add header
    sub = fill_header(h);
    PyDict_SetItemString(dict, "Header", sub);
    Py_DECREF(sub);

    b = h->current_block = h->blocklist;
    for (i = 0; i < h->nblocks; i++) {
        switch(b->blocktype) {
          case SDF_BLOCKTYPE_PLAIN_MESH:
          case SDF_BLOCKTYPE_POINT_MESH:
            setup_mesh(h, dict);
            break;
          case SDF_BLOCKTYPE_LAGRANGIAN_MESH:
            setup_lagrangian_mesh(h, dict);
            break;
          case SDF_BLOCKTYPE_PLAIN_VARIABLE:
          case SDF_BLOCKTYPE_POINT_VARIABLE:
            for (n = 0; n < b->ndims; n++) dims[n] = (int)b->dims[n];
            sdf_read_data(h);
            sub = PyArray_NewFromDescr(&PyArray_Type,
                PyArray_DescrFromType(typemap[b->datatype_out]), b->ndims,
                dims, NULL, b->data, NPY_F_CONTIGUOUS, NULL);
            PyDict_SetItemString(dict, b->name, sub);
            Py_DECREF(sub);
            break;
          case SDF_BLOCKTYPE_CONSTANT:
            switch(b->datatype) {
              case SDF_DATATYPE_REAL4:
                dd = *((float*)b->const_value);
                sub = PyFloat_FromDouble(dd);
                PyDict_SetItemString(dict, b->name, sub);
                Py_DECREF(sub);
                break;
              case SDF_DATATYPE_REAL8:
                dd = *((double*)b->const_value);
                sub = PyFloat_FromDouble(dd);
                PyDict_SetItemString(dict, b->name, sub);
                Py_DECREF(sub);
                break;
              case SDF_DATATYPE_INTEGER4:
                il = *((int32_t*)b->const_value);
                sub = PyLong_FromLong(il);
                PyDict_SetItemString(dict, b->name, sub);
                Py_DECREF(sub);
                break;
              case SDF_DATATYPE_INTEGER8:
                ll = *((int64_t*)b->const_value);
                sub = PyLong_FromLongLong(ll);
                PyDict_SetItemString(dict, b->name, sub);
                Py_DECREF(sub);
                break;
            }
            break;
         case SDF_BLOCKTYPE_STATION:
            extract_station_time_histories(h, kw, dict);
            break;
        }
        b = h->current_block = b->next;
    }

    return dict;
}


static PyMethodDef SDF_methods[] = {
    {"read", (PyCFunction)SDF_read, METH_VARARGS | METH_KEYWORDS,
        "Reads the SDF data and returns a dictionary of NumPy arrays" },
    {NULL}
};


static PyTypeObject SDF_type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "sdf.SDF",                 /* tp_name           */
    sizeof(SDFObject),         /* tp_basicsize      */
    0,                         /* tp_itemsize       */
    SDF_dealloc,               /* tp_dealloc */
    0,                         /* tp_print          */
    0,                         /* tp_getattr        */
    0,                         /* tp_setattr        */
    0,                         /* tp_compare        */
    0,                         /* tp_repr           */
    0,                         /* tp_as_number      */
    0,                         /* tp_as_sequence    */
    0,                         /* tp_as_mapping     */
    0,                         /* tp_hash           */
    0,                         /* tp_call           */
    0,                         /* tp_str            */
    0,                         /* tp_getattro       */
    0,                         /* tp_setattro       */
    0,                         /* tp_as_buffer      */
    Py_TPFLAGS_DEFAULT,        /* tp_flags          */
    "SDF constructor accepts two arguments.\n"
    "The first is the SDF filename to open. This argument is mandatory.\n"
    "The second argument is an optional integer. If it is non-zero then the\n"
    "data is converted from double precision to single.",  /* tp_doc      */
    0,                         /* tp_traverse       */
    0,                         /* tp_clear          */
    0,                         /* tp_richcompare    */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter           */
    0,                         /* tp_iternext       */
    SDF_methods,               /* tp_methods        */
    0,                         /* tp_members        */
    0,                         /* tp_getset         */
    0,                         /* tp_base           */
    0,                         /* tp_dict           */
    0,                         /* tp_descr_get      */
    0,                         /* tp_descr_set      */
    0,                         /* tp_dictoffset     */
    0,                         /* tp_init           */
    0,                         /* tp_alloc          */
    SDF_new,                   /* tp_new            */
};


#if PY_MAJOR_VERSION >= 3
  #define MOD_ERROR_VAL NULL
  #define MOD_SUCCESS_VAL(val) val
  #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
          static struct PyModuleDef moduledef = { \
            PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
          ob = PyModule_Create(&moduledef);
#else
  #define MOD_ERROR_VAL
  #define MOD_SUCCESS_VAL(val)
  #define MOD_INIT(name) void init##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
          ob = Py_InitModule3(name, methods, doc);
#endif


MOD_INIT(sdf)
{
    PyObject *m;

    MOD_DEF(m, "sdf", "SDF file reading library", NULL)

    if (m == NULL)
        return MOD_ERROR_VAL;

    if (PyType_Ready(&SDF_type) < 0)
        return MOD_ERROR_VAL;

    Py_INCREF(&SDF_type);
    if (PyModule_AddObject(m, "SDF", (PyObject *) &SDF_type) < 0)
        return MOD_ERROR_VAL;

    import_array();   /* required NumPy initialization */

    return MOD_SUCCESS_VAL(m);
}
