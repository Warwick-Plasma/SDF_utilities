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
    sdf_file_t *h;
} SDFObject;


typedef struct {
    PyObject_HEAD
    SDFObject *sdf;
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
    PyObject *label;
    PyObject *units;
    PyObject *array;
    SDFObject *sdf;
    sdf_block_t *b;
    int dim, ndims;
    npy_intp adims[4];
} Block;


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


static PyTypeObject SDFType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "sdf.SDF",                 /* tp_name           */
    sizeof(SDFObject),         /* tp_basicsize      */
    0,                         /* tp_itemsize       */
};


/*
 * Array type methods
 ******************************************************************************/

static PyObject *
Array_new(PyTypeObject *type, SDFObject *sdf, sdf_block_t *b)
{
    ArrayObject *self;

    self = (ArrayObject*)type->tp_alloc(type, 0);
    if (self) {
        self->sdf = sdf;
        self->b = b;
        self->mem = NULL;
        Py_INCREF(self->sdf);
    }

    return (PyObject*)self;
}


static void
Array_dealloc(PyObject *self)
{
    ArrayObject *ob = (ArrayObject*)self;
    if (!ob) return;
    if (ob->mem)
        free(ob->mem);
    else
        sdf_free_block_data(ob->sdf->h, ob->b);
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

static PyMemberDef BlockMesh_members[] = {
    {"label", T_OBJECT_EX, offsetof(Block, label), 0, "Axis label"},
    {"units", T_OBJECT_EX, offsetof(Block, units), 0, "Axis units"},
    {NULL}  /* Sentinel */
};


static PyObject *
Block_alloc(SDFObject *sdf, sdf_block_t *b)
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
    ob->sdf = sdf;
    ob->b = b;
    ob->dim = 0;
    ob->ndims = b->ndims;

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
                for (i=0; i < b->ndims; i++) {
                    ob->adims[i] = b->dims[i];
                    PyTuple_SetItem(ob->dims, i, PyLong_FromLong(b->dims[i]));
                }
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
    if (ob->label) {
        Py_XDECREF(ob->label);
    }
    if (ob->units) {
        Py_XDECREF(ob->units);
    }
    Py_DECREF(ob);

    return NULL;
}


static void
Block_dealloc(PyObject *self)
{
    Block *ob = (Block*)self;
    if (!ob) return;
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
    if (ob->label) {
        Py_XDECREF(ob->label);
    }
    if (ob->units) {
        Py_XDECREF(ob->units);
    }
    if (ob->data) {
        Py_XDECREF(ob->data);
    }
    self->ob_type->tp_free(self);
}


/*
 * SDF type methods
 ******************************************************************************/

static void
SDF_dealloc(PyObject* self)
{
    sdf_file_t *h = ((SDFObject*)self)->h;
    if (h) sdf_close(h);
    self->ob_type->tp_free(self);
}


static void setup_mesh(SDFObject *sdf, PyObject *dict, sdf_block_t *b)
{
    char *block_name = NULL;
    PyObject *array = NULL;
    PyObject *array2 = NULL;
    Block *block = NULL;
    Py_ssize_t n, i, ndims_dir;
    void *data = NULL;
    void *mem = NULL;
    size_t len_name, len_label, len_id;
    int add_grid = 0;

    if (!sdf->h || !b) return;

    sdf->h->current_block = b;
    sdf_read_data(sdf->h);
    if (b->grids) data = b->grids[0];
    block_name = b->name;
    len_name = strlen(b->name);
    len_id = strlen(b->id);

    if (!data) return;

    array = Array_new(&ArrayType, sdf, b);
    if (!array) goto free_mem;

    if (strncmp(b->id, "grid", len_id+1) == 0) add_grid = 1;

    for (n = 0; n < b->ndims; n++) {
        mem = block = NULL;
        ndims_dir = b->dims[n];

        len_label = strlen(b->dim_labels[n]);
        block_name = malloc(len_name + len_label + 2);
        if (!block_name) goto free_mem;

        memcpy(block_name, b->name, len_name);
        block_name[len_name] = '/';
        memcpy(block_name+len_name+1, b->dim_labels[n], len_label+1);

        /* Hack to node-centre the cartesian grid */
        if (add_grid) {
            ndims_dir--;
            if (b->datatype_out == SDF_DATATYPE_REAL4) {
                float v1, v2, *ptr_in, *ptr_out;
                ptr_in = b->grids[n];
                ptr_out = data = mem = malloc(ndims_dir * sizeof(*ptr_out));
                if (!mem) goto free_mem;
                for (i = 0; i < ndims_dir; i++) {
                    v1 = *ptr_in;
                    v2 = *(ptr_in+1);
                    *ptr_out++ = 0.5 * (v1 + v2);
                    ptr_in++;
                }
            } else {
                double v1, v2, *ptr_in, *ptr_out;
                ptr_in = b->grids[n];
                ptr_out = data = mem = malloc(ndims_dir * sizeof(*ptr_out));
                if (!mem) goto free_mem;
                for (i = 0; i < ndims_dir; i++) {
                    v1 = *ptr_in;
                    v2 = *(ptr_in+1);
                    *ptr_out++ = 0.5 * (v1 + v2);
                    ptr_in++;
                }
            }

            array2 = Array_new(&ArrayType, sdf, b);
            if (!array2) goto free_mem;
            ((ArrayObject*)array2)->mem = mem;

            block = (Block*)Block_alloc(sdf, b);
            if (!block) goto free_mem;

            block->ndims = 1;
            PyTuple_SetItem(block->dims, 0, PyLong_FromLongLong(ndims_dir));
            block->adims[0] = ndims_dir;

            block->array = array2;

            block->label = PyString_FromString(b->dim_labels[n]);
            if (block->label == NULL) goto free_mem;

            block->units = PyString_FromString(b->dim_units[n]);
            if (block->units == NULL) goto free_mem;

            block->data = PyArray_NewFromDescr(&PyArray_Type,
                PyArray_DescrFromType(typemap[b->datatype_out]), block->ndims,
                block->adims, NULL, data, NPY_ARRAY_F_CONTIGUOUS, NULL);
            if (!block->data) goto free_mem;

            PyArray_SetBaseObject((PyArrayObject*)block->data, block->array);
            PyDict_SetItemString(dict, block_name, (PyObject*)block);
            Py_DECREF(block);
            free(block_name);

            /* Now add the original grid with "_node" appended */
            ndims_dir++;

            block_name = malloc(len_name + len_label + 2 + 5);
            if (!block_name) goto free_mem;

            memcpy(block_name, b->name, len_name);
            memcpy(block_name+len_name, "_node", 5);
            block_name[len_name+5] = '/';
            memcpy(block_name+len_name+6, b->dim_labels[n], len_label+1);
        }

        data = b->grids[n];

        block = (Block*)Block_alloc(sdf, b);
        if (!block) goto free_mem;

        block->ndims = 1;
        PyTuple_SetItem(block->dims, 0, PyLong_FromLongLong(ndims_dir));
        block->adims[0] = ndims_dir;

        block->array = array;

        block->label = PyString_FromString(b->dim_labels[n]);
        if (block->label == NULL) goto free_mem;

        block->units = PyString_FromString(b->dim_units[n]);
        if (block->units == NULL) goto free_mem;

        block->data = PyArray_NewFromDescr(&PyArray_Type,
            PyArray_DescrFromType(typemap[b->datatype_out]), block->ndims,
            block->adims, NULL, data, NPY_ARRAY_F_CONTIGUOUS, NULL);
        if (!block->data) goto free_mem;

        PyArray_SetBaseObject((PyArrayObject*)block->data, block->array);
        PyDict_SetItemString(dict, block_name, (PyObject*)block);
        Py_DECREF(block);
        Py_INCREF(block->array);
        free(block_name);
    }
    Py_DECREF(block->array);

    return;

free_mem:
    if (block_name) free(block_name);
    if (mem) free(mem);
    if (block) Py_DECREF(block);
    if (array) Py_DECREF(array);
    if (array2) Py_DECREF(array2);
    sdf_free_block_data(sdf->h, b);
}


static void setup_lagrangian_mesh(SDFObject *sdf, PyObject *dict,
                                  sdf_block_t *b)
{
    char *block_name = NULL;
    PyObject *array = NULL;
    Block *block = NULL;
    Py_ssize_t n;
    void *data = NULL;
    size_t len_name, len_label;

    if (!sdf->h || !b) return;

    sdf->h->current_block = b;
    sdf_read_data(sdf->h);
    if (b->grids) data = b->grids[0];
    block_name = b->name;
    len_name = strlen(b->name);

    if (!data) return;

    array = Array_new(&ArrayType, sdf, b);
    if (!array) goto free_mem;

    for (n = 0; n < b->ndims; n++) {
        block = NULL;

        len_label = strlen(b->dim_labels[n]);
        block_name = malloc(len_name + len_label + 2);
        if (!block_name) goto free_mem;

        memcpy(block_name, b->name, len_name);
        block_name[len_name] = '/';
        memcpy(block_name+len_name+1, b->dim_labels[n], len_label+1);

        data = b->grids[n];

        block = (Block*)Block_alloc(sdf, b);
        if (!block) goto free_mem;

        block->array = array;

        block->label = PyString_FromString(b->dim_labels[n]);
        if (block->label == NULL) goto free_mem;

        block->units = PyString_FromString(b->dim_units[n]);
        if (block->units == NULL) goto free_mem;

        block->data = PyArray_NewFromDescr(&PyArray_Type,
            PyArray_DescrFromType(typemap[b->datatype_out]), block->ndims,
            block->adims, NULL, data, NPY_ARRAY_F_CONTIGUOUS, NULL);
        if (!block->data) goto free_mem;

        PyArray_SetBaseObject((PyArrayObject*)block->data, block->array);
        PyDict_SetItemString(dict, block_name, (PyObject*)block);
        Py_DECREF(block);
        Py_INCREF(block->array);
        free(block_name);
    }
    Py_DECREF(block->array);

    return;

free_mem:
    if (block_name) free(block_name);
    if (block) Py_DECREF(block);
    if (array) Py_DECREF(array);
    sdf_free_block_data(sdf->h, b);
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
setup_array(SDFObject *sdf, PyObject *dict, sdf_block_t *b)
{
    char *block_name = NULL;
    PyObject *array = NULL;
    Block *block = NULL;
    void *data = NULL;

    if (!sdf->h || !b) return NULL;

    sdf->h->current_block = b;
    sdf_read_data(sdf->h);
    data = b->data;
    block_name = b->name;

    if (!data) return NULL;

    array = Array_new(&ArrayType, sdf, b);
    if (!array) goto free_mem;

    block = (Block*)Block_alloc(sdf, b);
    if (!block) goto free_mem;

    block->array = array;

    if (b->units) {
        block->units = PyString_FromString(b->units);
        if (block->units == NULL) goto free_mem;
    }

    block->data = PyArray_NewFromDescr(&PyArray_Type,
        PyArray_DescrFromType(typemap[b->datatype_out]), block->ndims,
        block->adims, NULL, data, NPY_ARRAY_F_CONTIGUOUS, NULL);
    if (!block->data) goto free_mem;

    PyArray_SetBaseObject((PyArrayObject*)block->data, block->array);
    PyDict_SetItemString(dict, block_name, (PyObject*)block);
    Py_DECREF(block);

    return (PyObject*)block;

free_mem:
    if (block) Py_DECREF(block);
    if (array) Py_DECREF(array);
    sdf_free_block_data(sdf->h, b);
    return NULL;
}


static PyObject *
setup_constant(SDFObject *sdf, PyObject *dict, sdf_block_t *b)
{
    Block *block = NULL;
    double dd;
    long il;
    long long ll;

    block = (Block*)Block_alloc(sdf, b);
    if (!block) return NULL;

    switch(b->datatype) {
        case SDF_DATATYPE_REAL4:
            dd = *((float*)b->const_value);
            block->data = PyFloat_FromDouble(dd);
            break;
        case SDF_DATATYPE_REAL8:
            dd = *((double*)b->const_value);
            block->data = PyFloat_FromDouble(dd);
            break;
        case SDF_DATATYPE_INTEGER4:
            il = *((int32_t*)b->const_value);
            block->data = PyLong_FromLong(il);
            break;
        case SDF_DATATYPE_INTEGER8:
            ll = *((int64_t*)b->const_value);
            block->data = PyLong_FromLongLong(ll);
            break;
    }

    PyDict_SetItemString(dict, b->name, (PyObject*)block);

    Py_DECREF(block);

    return (PyObject*)block;
}


static PyObject* SDF_read(PyObject *self, PyObject *args, PyObject *kw)
{
    SDFObject *sdf;
    PyTypeObject *type = &SDFType;
    sdf_file_t *h;
    sdf_block_t *b;
    PyObject *dict, *sub;
    int i, convert, use_mmap, mode;
    comm_t comm;
    const char *file;
    static char *kwlist[] = {"file", "convert", "mmap", "stations",
        "variables", "t0", "t1", NULL};
    PyObject *stations = NULL, *variables = NULL;
    double t0 = -DBL_MAX, t1 = DBL_MAX;

    convert = 0; use_mmap = 1; mode = SDF_READ; comm = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "s|iiO!O!dd", kwlist, &file,
            &convert, &use_mmap, &PyList_Type, &stations, &PyList_Type,
            &variables, &t0, &t1))
        return NULL;

    sdf = (SDFObject*)type->tp_alloc(type, 0);
    if (sdf == NULL) {
        PyErr_Format(PyExc_MemoryError, "Failed to allocate SDF object");
        return NULL;
    }

    h = sdf_open(file, comm, mode, use_mmap);
    sdf->h = h;
    if (!sdf->h) {
        PyErr_Format(PyExc_IOError, "Failed to open file: '%s'", file);
        Py_DECREF(sdf);
        return NULL;
    }

    if (convert) h->use_float = 1;

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
                setup_mesh(sdf, dict, b);
                break;
            case SDF_BLOCKTYPE_LAGRANGIAN_MESH:
                setup_lagrangian_mesh(sdf, dict, b);
                break;
            case SDF_BLOCKTYPE_PLAIN_VARIABLE:
            case SDF_BLOCKTYPE_POINT_VARIABLE:
            case SDF_BLOCKTYPE_ARRAY:
                setup_array(sdf, dict, b);
                break;
            case SDF_BLOCKTYPE_CONSTANT:
                setup_constant(sdf, dict, b);
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

    Py_DECREF(sdf);

    return dict;
}


static PyMethodDef SDF_methods[] = {
    {"read", (PyCFunction)SDF_read, METH_VARARGS | METH_KEYWORDS,
     "read(file, [convert, mmap, stations, variables, t0, t1])\n\n"
     "Reads the SDF data and returns a dictionary of NumPy arrays.\n\n"
     "Parameters\n"
     "----------\n"
     "file : string\n"
     "    The name of the SDF file to open.\n"
     "convert : bool, optional\n"
     "    Convert double precision data to single when reading file.\n"
     "mmap : bool, optional\n"
     "    Use mmap to map file contents into memory.\n"
     "stations : string list, optional\n"
     "    List of stations to read.\n"
     "variables : string list, optional\n"
     "    List of station variables to read.\n"
     "t0 : double, optional\n"
     "    Starting time for station data.\n"
     "t1 : double, optional\n"
     "    Ending time for station data.\n"
     },
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

    MOD_DEF(m, "sdf", "SDF file reading library", SDF_methods)

    if (m == NULL)
        return MOD_ERROR_VAL;

    PyModule_AddStringConstant(m, "__version__", "2.0.0");

    SDFType.tp_dealloc = SDF_dealloc;
    SDFType.tp_flags = Py_TPFLAGS_DEFAULT;
    SDFType.tp_doc = "SDF constructor accepts two arguments.\n"
        "The first is the SDF filename to open. This argument is mandatory.\n"
        "The second argument is an optional integer. If it is non-zero then "
        "the\ndata is converted from double precision to single.";
    SDFType.tp_methods = SDF_methods;
    if (PyType_Ready(&SDFType) < 0)
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

    BlockType.tp_name = "sdf.Block";
    BlockType.tp_dealloc = Block_dealloc;
    BlockType.tp_members = Block_members;
    if (PyType_Ready(&BlockType) < 0)
        return MOD_ERROR_VAL;
    Py_INCREF(&BlockType);
    if (PyModule_AddObject(m, "Block", (PyObject *)&BlockType) < 0)
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
    BlockBase.tp_members = BlockMesh_members;

    ADD_TYPE(BlockPlainMesh, BlockBase);
    ADD_TYPE(BlockPointMesh, BlockBase);
    ADD_TYPE(BlockLagrangianMesh, BlockBase);

    import_array();   /* required NumPy initialization */

    return MOD_SUCCESS_VAL(m);
}
