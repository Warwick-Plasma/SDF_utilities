#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <float.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <structmember.h>
#include "sdf.h"

/* Backwards compatibility */

#if PY_MAJOR_VERSION < 3
    #define PyInt_FromLong PyLong_FromLong
#endif

#ifndef NPY_ARRAY_F_CONTIGUOUS
    #define NPY_ARRAY_F_CONTIGUOUS NPY_F_CONTIGUOUS
#endif

#ifndef PyVarObject_HEAD_INIT
    #define PyVarObject_HEAD_INIT(type, size) \
        PyObject_HEAD_INIT(type) size,
#endif

#ifndef PyArray_SetBaseObject
    #define PyArray_SetBaseObject(array, base) \
             PyArray_BASE(array) = base
#endif

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

int sdf_free_block_data(sdf_file_t *h, sdf_block_t *b);

static const int typemap[] = {
    0,
    NPY_UINT32,
    NPY_UINT64,
    NPY_FLOAT,
    NPY_DOUBLE,
#ifdef NPY_FLOAT128
    NPY_FLOAT128,
#else
    0,
#endif
    NPY_CHAR,
    NPY_CHAR,
};


typedef struct {
    PyObject_HEAD
    PyObject *sdf;
    sdf_file_t *h;
    sdf_block_t *b;
    void *mem;
} ArrayObject;


typedef struct {
    PyObject_HEAD
    PyObject *id;
    PyObject *name;
    PyObject *data_length;
    PyObject *datatype;
    PyObject *dims;
    PyObject *data;
    sdf_file_t *h;
    sdf_block_t *b;
} Block;


typedef struct {
    PyObject_HEAD
    sdf_file_t *h;
} SDFObject;


static PyTypeObject ArrayType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "Array",                   /* tp_name           */
    sizeof(ArrayObject),       /* tp_basicsize      */
    0,                         /* tp_itemsize       */
};


static PyTypeObject BlockType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "sdf.Block",               /* tp_name           */
    sizeof(Block),             /* tp_basicsize      */
    0,                         /* tp_itemsize       */
};


static PyTypeObject BlockBase;
static PyTypeObject BlockMeshType;
static PyTypeObject BlockPlainMeshType;
static PyTypeObject BlockPointMeshType;
static PyTypeObject BlockLagrangianMeshType;
static PyTypeObject BlockPlainVariableType;
static PyTypeObject BlockPointVariableType;
static PyTypeObject BlockArrayType;
static PyTypeObject BlockConstantType;
static PyTypeObject BlockStationType;
static PyTypeObject BlockStitchedMaterialType;


static PyTypeObject SDF_type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "sdf.SDF",                 /* tp_name           */
    sizeof(SDFObject),         /* tp_basicsize      */
    0,                         /* tp_itemsize       */
};


/*
 * Array type methods
 ******************************************************************************/

static PyObject *
Array_new(PyTypeObject *type, PyObject *sdf, sdf_file_t *h, sdf_block_t *b)
{
    PyObject *self;
    ArrayObject *ob;

    self = type->tp_alloc(type, 0);
    if (self) {
        ob = (ArrayObject*)self;
        ob->sdf = sdf;
        ob->h = h;
        ob->b = b;
        ob->mem = NULL;
        Py_INCREF(sdf);
    }

    return self;
}


static void
Array_dealloc(PyObject *self)
{
    ArrayObject *ob = (ArrayObject*)self;
    if (!self) return;
    if (ob->mem)
        free(ob->mem);
    else
        sdf_free_block_data(ob->h, ob->b);
    Py_XDECREF(ob->sdf);
    self->ob_type->tp_free(self);
}


/*
 * Block type methods
 ******************************************************************************/

static PyMemberDef Block_members[] = {
    {"id", T_OBJECT_EX, offsetof(Block, id), 0, "Block id"},
    {"name", T_OBJECT_EX, offsetof(Block, name), 0, "Block name"},
    {"data_length", T_OBJECT_EX, offsetof(Block, data_length), 0, "Data size"},
    {"datatype", T_OBJECT_EX, offsetof(Block, datatype), 0, "Data type"},
    {"dims", T_OBJECT_EX, offsetof(Block, dims), 0, "Data dimensions"},
    {"data", T_OBJECT_EX, offsetof(Block, data), 0, "Block data contents"},
    {NULL}  /* Sentinel */
};


static PyObject *
Block_alloc(sdf_file_t *h, sdf_block_t *b)
{
    Block *ob;
    PyTypeObject *type;
    Py_ssize_t i;

    switch(b->blocktype) {
        case SDF_BLOCKTYPE_PLAIN_MESH:
            type = &BlockPlainMeshType;
            break;
        case SDF_BLOCKTYPE_POINT_MESH:
            type = &BlockPointMeshType;
            break;
        case SDF_BLOCKTYPE_LAGRANGIAN_MESH:
            type = &BlockLagrangianMeshType;
            break;
        case SDF_BLOCKTYPE_PLAIN_VARIABLE:
            type = &BlockPlainVariableType;
            break;
        case SDF_BLOCKTYPE_POINT_VARIABLE:
            type = &BlockPointVariableType;
            break;
        case SDF_BLOCKTYPE_ARRAY:
            type = &BlockArrayType;
            break;
        case SDF_BLOCKTYPE_CONSTANT:
            type = &BlockConstantType;
            break;
        case SDF_BLOCKTYPE_STATION:
            type = &BlockStationType;
            break;
        case SDF_BLOCKTYPE_STITCHED_MATERIAL:
            type = &BlockStitchedMaterialType;
            break;
        default:
            type = &BlockType;
    }

    ob = (Block *)type->tp_alloc(type, 0);
    ob->h = h;
    ob->b = b;

    if (b->id) {
        ob->id = PyString_FromString(b->id);
        if (ob->id == NULL) goto error;
    }

    if (b->name) {
        ob->name = PyString_FromString(b->name);
        if (ob->name == NULL) goto error;
    }

    if (b->data_length) {
        ob->data_length = PyLong_FromLongLong(b->data_length);
        if (ob->data_length == NULL) goto error;
    }

    if (b->datatype_out) {
        ob->datatype = PyArray_TypeObjectFromType(typemap[b->datatype_out]);
        if (ob->datatype == NULL) goto error;
    }

    if (b->ndims) {
        if (b->blocktype == SDF_BLOCKTYPE_PLAIN_MESH)
            ob->dims = PyTuple_New(1);
        else
            ob->dims = PyTuple_New(b->ndims);
        if (ob->dims == NULL) goto error;
    }

    switch(b->blocktype) {
        case SDF_BLOCKTYPE_PLAIN_MESH:
            break;
        case SDF_BLOCKTYPE_POINT_MESH:
            break;
        case SDF_BLOCKTYPE_LAGRANGIAN_MESH:
        case SDF_BLOCKTYPE_PLAIN_VARIABLE:
        case SDF_BLOCKTYPE_POINT_VARIABLE:
        case SDF_BLOCKTYPE_ARRAY:
            if (b->dims && ob->dims) {
                for (i=0; i < b->ndims; i++)
                    PyTuple_SetItem(ob->dims, i, PyLong_FromLong(b->dims[i]));
            }
            break;
        case SDF_BLOCKTYPE_CONSTANT:
            PyTuple_SetItem(ob->dims, 0, PyLong_FromLong(1));
            break;
        case SDF_BLOCKTYPE_STATION:
            break;
        case SDF_BLOCKTYPE_STITCHED_MATERIAL:
            break;
    }

    return (PyObject *)ob;

error:
    if (ob->id) {
        Py_DECREF(ob->id);
    }
    if (ob->name) {
        Py_DECREF(ob->name);
    }
    if (ob->data_length) {
        Py_DECREF(ob->data_length);
    }
    if (ob->datatype) {
        Py_DECREF(ob->datatype);
    }
    if (ob->dims) {
        Py_DECREF(ob->dims);
    }
    Py_DECREF(ob);

    return NULL;
}


static void
Block_dealloc(PyObject *self)
{
    Block *ob = (Block*)self;
    if (!self) return;
    if (ob->id) {
        Py_XDECREF(ob->id);
    }
    if (ob->name) {
        Py_XDECREF(ob->name);
    }
    if (ob->data_length) {
        Py_XDECREF(ob->data_length);
    }
    if (ob->datatype) {
        Py_XDECREF(ob->datatype);
    }
    if (ob->dims) {
        Py_XDECREF(ob->dims);
    }
    if (ob->data) {
        Py_XDECREF(ob->data);
    }
    self->ob_type->tp_free(self);
}


/*
 * SDF type methods
 ******************************************************************************/

static int convert, use_mmap, mode;
static comm_t comm;

static PyObject *
SDF_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    sdf_file_t *h;
    const char *file;
    static char *kwlist[] = {"file", "convert", "mmap", NULL};
    SDFObject *self;

    convert = 0; use_mmap = 1; mode = SDF_READ; comm = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|ii", kwlist, &file,
        &convert, &use_mmap)) return NULL;

    self = (SDFObject*)type->tp_alloc(type, 0);
    if (self == NULL) {
        PyErr_Format(PyExc_MemoryError, "Failed to allocate SDF object");
        return NULL;
    }

    h = sdf_open(file, comm, mode, use_mmap);
    self->h = h;
    if (!self->h) {
        PyErr_Format(PyExc_IOError, "Failed to open file: '%s'", file);
        Py_DECREF(self);
        return NULL;
    }

    if (convert) h->use_float = 1;

    return (PyObject*)self;
}


static void
SDF_dealloc(PyObject* self)
{
    sdf_file_t *h = ((SDFObject*)self)->h;
    if (h) sdf_close(h);
    self->ob_type->tp_free(self);
}


static void setup_mesh(PyObject *self, PyObject *dict)
{
    sdf_file_t *h = ((SDFObject*)self)->h;
    sdf_block_t *b = h->current_block;
    Py_ssize_t i, n, ndims;
    size_t l1, l2;
    char *label = NULL;
    void *grid, *grid_ptr = NULL;
    PyObject *sub = NULL, *array = NULL, *array2 = NULL, *block = NULL;
    npy_intp dims[3] = {0,0,0};

    if (!h || !b) return;

    sdf_read_data(h);
    if (!b->grids || !b->grids[0]) return;

    array = Array_new(&ArrayType, self, h, b);
    if (!array) goto free_mem;

    for (n = 0; n < b->ndims; n++) {
        sub = array2 = block = grid_ptr = NULL;
        ndims = b->dims[n];

        l1 = strlen(b->name);
        l2 = strlen(b->dim_labels[n]);
        label = malloc(l1 + l2 + 2);
        if (!label) return;

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
                grid_ptr = grid = out = malloc(ndims * sizeof(*out));
                if (!grid_ptr) goto free_mem;
                for (i = 0; i < ndims; i++) {
                    v1 = *ptr;
                    v2 = *(ptr+1);
                    *out++ = 0.5 * (v1 + v2);
                    ptr++;
                }
            } else {
                double v1, v2, *ptr, *out;
                ptr = grid;
                grid_ptr = grid = out = malloc(ndims * sizeof(*out));
                if (!grid_ptr) goto free_mem;
                for (i = 0; i < ndims; i++) {
                    v1 = *ptr;
                    v2 = *(ptr+1);
                    *out++ = 0.5 * (v1 + v2);
                    ptr++;
                }
            }

            dims[0] = ndims;
            array2 = Array_new(&ArrayType, self, h, b);
            if (!array2) goto free_mem;
            ((ArrayObject*)array2)->mem = grid;

            block = Block_alloc(h, b);
            if (!block) goto free_mem;

            PyTuple_SetItem(((Block*)block)->dims, 0,
                PyLong_FromLongLong(ndims));

            sub = PyArray_NewFromDescr(&PyArray_Type,
                PyArray_DescrFromType(typemap[b->datatype_out]), 1,
                dims, NULL, grid, NPY_ARRAY_F_CONTIGUOUS, NULL);
            if (!sub) goto free_mem;

            PyArray_SetBaseObject((PyArrayObject*)sub, array2);
            PyDict_SetItemString(dict, label, block);
            ((Block*)block)->data = sub;
            Py_DECREF(block);

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

        block = Block_alloc(h, b);
        if (!block) goto free_mem;

        PyTuple_SetItem(((Block*)block)->dims, 0, PyLong_FromLongLong(ndims));

        sub = PyArray_NewFromDescr(&PyArray_Type,
            PyArray_DescrFromType(typemap[b->datatype_out]), 1,
            dims, NULL, grid, NPY_ARRAY_F_CONTIGUOUS, NULL);
        if (!sub) goto free_mem;

        Py_INCREF(array);
        PyArray_SetBaseObject((PyArrayObject*)sub, array);
        PyDict_SetItemString(dict, label, block);
        ((Block*)block)->data = sub;
        Py_DECREF(block);
    }
    Py_DECREF(array);

    return;

free_mem:
    if (label) free(label);
    if (grid_ptr) free(grid_ptr);
    if (block) Py_DECREF(block);
    if (array) Py_DECREF(array);
    if (array2) Py_DECREF(array2);
    sdf_free_block_data(h, b);
}


static void setup_lagrangian_mesh(PyObject *self, PyObject *dict)
{
    sdf_file_t *h = ((SDFObject*)self)->h;
    sdf_block_t *b = h->current_block;
    int n;
    size_t l1, l2;
    char *label = NULL;
    void *grid;
    PyObject *sub = NULL, *array = NULL, *block = NULL;
    npy_intp dims[3] = {0,0,0};

    if (!h || !b) return;

    sdf_read_data(h);
    if (!b->grids || !b->grids[0]) return;

    for (n = 0; n < b->ndims; n++) dims[n] = (int)b->dims[n];

    array = Array_new(&ArrayType, self, h, b);
    if (!array) goto free_mem;

    for (n = 0; n < b->ndims; n++) {
        sub = block = NULL;

        l1 = strlen(b->name);
        l2 = strlen(b->dim_labels[n]);
        label = malloc(l1 + l2 + 2);
        if (!label) goto free_mem;
        memcpy(label, b->name, l1);
        label[l1] = '/';
        memcpy(label+l1+1, b->dim_labels[n], l2+1);

        l1 = strlen(b->id);
        grid = b->grids[n];

        block = Block_alloc(h, b);
        if (!block) goto free_mem;
        sub = PyArray_NewFromDescr(&PyArray_Type,
            PyArray_DescrFromType(typemap[b->datatype_out]), b->ndims,
            dims, NULL, grid, NPY_ARRAY_F_CONTIGUOUS, NULL);
        if (!sub) goto free_mem;

        Py_INCREF(array);
        PyArray_SetBaseObject((PyArrayObject*)sub, array);
        PyDict_SetItemString(dict, label, block);
        ((Block*)block)->data = sub;
        Py_DECREF(block);
    }
    Py_DECREF(array);

    return;

free_mem:
    if (label) free(label);
    if (block) Py_DECREF(block);
    if (array) Py_DECREF(array);
    sdf_free_block_data(h, b);
}


static void extract_station_time_histories(sdf_file_t *h, PyObject *stations,
      PyObject *variables, double t0, double t1, PyObject *dict)
{
    Py_ssize_t nvars, i, nstat;
    PyObject *sub;
    char **var_names, *timehis, *v, *key;
    long *stat, ii;
    int *size, *offset, nrows, row_size;
    sdf_block_t *b;
    npy_intp dims[1];

    if ( !stations ) {
        nstat = 1;
        stat = (long *)malloc(sizeof(long));
        stat[0] = 0;
    } else {
        /* Force 'stat' to be valid input for sdf_read_station_timehis */
        nstat = PyList_Size(stations);
        stat = (long *)calloc(nstat, sizeof(long));
        nstat = 0;
        for ( ii=0; ii<h->current_block->nstations; ii++ ) {
            sub = PyInt_FromLong(ii+1);
            if ( PySequence_Contains(stations, sub) ) {
                stat[nstat] = ii;
                nstat++;
            }
            Py_DECREF(sub);
        }
    }

    if ( !nstat ) {
        free(stat);
        return;
    }

    if ( !variables ) {
        free(stat);
        return;
    }

    nvars = PyList_Size(variables);
    if ( !nvars ) {
        free(stat);
        return;
    }

    var_names = (char **)malloc(nvars*sizeof(char *));
    for ( i=0; i<nvars; i++ ) {
        sub = PyList_GetItem(variables, i);
        var_names[i] = PyString_AsString(sub);
        if ( !var_names[i] ) {
            free(var_names);
            free(stat);
            PyErr_SetString(PyExc_TypeError,
                    "'variables' keyword must be a string or list of strings");
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
    key = malloc(3*h->string_length+3);
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

        sprintf(key, "%s/%s/%s", b->name,
                b->station_names[stat[(int)(i-1)/nvars]],
                var_names[(i-1)%nvars]);

        PyDict_SetItemString(dict, key, sub);
        Py_DECREF(sub);

        v += nrows * size[i];
    }

    free(var_names);
    free(size);
    free(key);
    free(stat);
    free(offset);
}


int append_station_metadata(sdf_block_t *b, PyObject *dict)
{
    PyObject *block, *station, *variable;
    int i;
    Py_ssize_t j;

    /* Sanity check */
    if ( !PyDict_Check(dict) )
        return -1;

    block = PyDict_New();
    PyDict_SetItemString(dict, b->name, block);

    for ( i=0; i<b->nstations; i++ ) {
        station = PyList_New(b->station_nvars[i]);

        for ( j=0; j<b->station_nvars[i]; j++ ) {
            variable = PyString_FromString(b->material_names[i+j+1]);
            PyList_SET_ITEM(station, j, variable);
        }

        PyDict_SetItemString(block, b->station_names[i], station);
        Py_DECREF(station);
    }

    Py_DECREF(block);

    return 0;
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


static PyObject *material_names(sdf_block_t *b)
{
    PyObject *matnames = PyList_New(b->ndims), *name;
    Py_ssize_t i;

    for ( i=0; i<b->ndims; i++ ) {
        name = PyString_FromString(b->material_names[i]);
        PyList_SET_ITEM(matnames, i, name);
    }

    return matnames;
}


static PyObject *
setup_array(PyObject *self, PyObject *dict, sdf_file_t *h, sdf_block_t *b)
{
    int n;
    npy_intp dims[3] = {0,0,0};
    PyObject *sub = NULL, *array = NULL, *block = NULL;

    if (!h || !b) return NULL;

    sdf_read_data(h);
    if (!b->data) return NULL;

    for (n = 0; n < b->ndims; n++) dims[n] = (int)b->dims[n];

    array = Array_new(&ArrayType, self, h, b);
    if (!array) goto free_mem;

    block = Block_alloc(h, b);
    if (!block) goto free_mem;
    sub = PyArray_NewFromDescr(&PyArray_Type,
        PyArray_DescrFromType(typemap[b->datatype_out]), b->ndims,
        dims, NULL, b->data, NPY_ARRAY_F_CONTIGUOUS, NULL);
    if (!sub) goto free_mem;

    PyArray_SetBaseObject((PyArrayObject*)sub, array);
    PyDict_SetItemString(dict, b->name, block);
    ((Block*)block)->data = sub;
    Py_DECREF(block);

    return block;

free_mem:
    if (block) Py_DECREF(block);
    if (array) Py_DECREF(array);
    sdf_free_block_data(h, b);
    return NULL;
}


static PyObject *
setup_constant(PyObject *self, PyObject *dict, sdf_file_t *h, sdf_block_t *b)
{
    PyObject *sub = NULL, *block = NULL;
    double dd;
    long il;
    long long ll;

    block = Block_alloc(h, b);
    if (!block) return NULL;

    switch(b->datatype) {
        case SDF_DATATYPE_REAL4:
            dd = *((float*)b->const_value);
            sub = PyFloat_FromDouble(dd);
            break;
        case SDF_DATATYPE_REAL8:
            dd = *((double*)b->const_value);
            sub = PyFloat_FromDouble(dd);
            break;
        case SDF_DATATYPE_INTEGER4:
            il = *((int32_t*)b->const_value);
            sub = PyLong_FromLong(il);
            break;
        case SDF_DATATYPE_INTEGER8:
            ll = *((int64_t*)b->const_value);
            sub = PyLong_FromLongLong(ll);
            break;
    }

    PyDict_SetItemString(dict, b->name, block);
    ((Block*)block)->data = sub;

    Py_DECREF(block);

    return block;
}


static PyObject* SDF_read(PyObject *self, PyObject *args, PyObject *kw)
{
    SDFObject *ob = (SDFObject*)self;
    sdf_file_t *h;
    sdf_block_t *b;
    PyObject *dict, *sub;
    int i;

    static char *kwlist[] = {"stations", "variables", "t0", "t1", NULL};
    PyObject *stations = NULL, *variables = NULL;
    double t0 = -DBL_MAX, t1 = DBL_MAX;

    if ( !PyArg_ParseTupleAndKeywords(args, kw, "|O!O!dd", kwlist,
            &PyList_Type, &stations, &PyList_Type, &variables, &t0, &t1) )
        return NULL;

    h = ob->h;

    /* Close file and re-open it if it has already been read */
    if (h->blocklist) {
        h = sdf_open(h->filename, comm, mode, use_mmap);
        sdf_close(ob->h);
        ob->h = h;
        if (!ob->h) {
            PyErr_Format(PyExc_IOError, "Failed to open file: '%s'",
                    h->filename);
            Py_DECREF(self);
            return NULL;
        }
    }

    sdf_read_blocklist(h);
    dict = PyDict_New();

    /* Add header */
    sub = fill_header(h);
    PyDict_SetItemString(dict, "Header", sub);
    Py_DECREF(sub);

    b = h->current_block = h->blocklist;
    for (i = 0; i < h->nblocks; i++) {
        switch(b->blocktype) {
            case SDF_BLOCKTYPE_PLAIN_MESH:
            case SDF_BLOCKTYPE_POINT_MESH:
                setup_mesh(self, dict);
                break;
            case SDF_BLOCKTYPE_LAGRANGIAN_MESH:
                setup_lagrangian_mesh(self, dict);
                break;
            case SDF_BLOCKTYPE_PLAIN_VARIABLE:
            case SDF_BLOCKTYPE_POINT_VARIABLE:
            case SDF_BLOCKTYPE_ARRAY:
                setup_array(self, dict, h, b);
                break;
            case SDF_BLOCKTYPE_CONSTANT:
                setup_constant(self, dict, h, b);
                break;
            case SDF_BLOCKTYPE_STATION:
                sub = PyDict_GetItemString(dict, "StationBlocks");
                if ( !sub ) {
                    sub = PyDict_New();
                    PyDict_SetItemString(dict, "StationBlocks", sub);
                }
                append_station_metadata(b, sub);
                extract_station_time_histories(h, stations, variables, t0, t1,
                        dict);
                break;
            case SDF_BLOCKTYPE_STITCHED_MATERIAL:
                sub = material_names(b);
                PyDict_SetItemString(dict, "Materials", sub);
                Py_DECREF(sub);
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

#define ADD_TYPE(name,base) do { \
        name##Type = base; \
        name##Type.tp_name = "sdf." #name; \
        if (PyType_Ready(&name##Type) < 0) \
            return MOD_ERROR_VAL; \
        Py_INCREF(&name##Type); \
        if (PyModule_AddObject(m, #name, (PyObject *)&name##Type) < 0) \
            return MOD_ERROR_VAL; \
    } while(0)


MOD_INIT(sdf)
{
    PyObject *m;

    MOD_DEF(m, "sdf", "SDF file reading library", NULL)

    if (m == NULL)
        return MOD_ERROR_VAL;

    SDF_type.tp_dealloc = SDF_dealloc;
    SDF_type.tp_flags = Py_TPFLAGS_DEFAULT;
    SDF_type.tp_doc = "SDF constructor accepts two arguments.\n"
        "The first is the SDF filename to open. This argument is mandatory.\n"
        "The second argument is an optional integer. If it is non-zero then "
        "the\ndata is converted from double precision to single.";
    SDF_type.tp_methods = SDF_methods;
    SDF_type.tp_new = SDF_new;
    if (PyType_Ready(&SDF_type) < 0)
        return MOD_ERROR_VAL;

    ArrayType.tp_dealloc = Array_dealloc;
    ArrayType.tp_flags = Py_TPFLAGS_DEFAULT;
    if (PyType_Ready(&ArrayType) < 0)
        return MOD_ERROR_VAL;

    BlockType.tp_flags = Py_TPFLAGS_DEFAULT;
    BlockType.tp_doc = "SDF block type.\n"
        "Contains the data and metadata for a single "
        "block from an SDF file.";
    BlockBase = BlockType;
    BlockBase.tp_base = &BlockType;

    BlockType.tp_name = "BlockType";
    BlockType.tp_dealloc = Block_dealloc;
    BlockType.tp_members = Block_members;
    if (PyType_Ready(&BlockType) < 0)
        return MOD_ERROR_VAL;

    ADD_TYPE(BlockConstant, BlockBase);
    ADD_TYPE(BlockStation, BlockBase);
    ADD_TYPE(BlockStitchedMaterial, BlockBase);
    ADD_TYPE(BlockArray, BlockBase);
    ADD_TYPE(BlockMesh, BlockBase);

    BlockBase.tp_base = &BlockArrayType;

    ADD_TYPE(BlockPlainVariable, BlockBase);
    ADD_TYPE(BlockPointVariable, BlockBase);

    BlockBase.tp_base = &BlockMeshType;

    ADD_TYPE(BlockPlainMesh, BlockBase);
    ADD_TYPE(BlockPointMesh, BlockBase);
    ADD_TYPE(BlockLagrangianMesh, BlockBase);

    Py_INCREF(&SDF_type);
    if (PyModule_AddObject(m, "SDF", (PyObject *) &SDF_type) < 0)
        return MOD_ERROR_VAL;

    import_array();   /* required NumPy initialization */

    return MOD_SUCCESS_VAL(m);
}
