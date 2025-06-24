import pytest
import warnings
from visualkeras.layer_utils import extract_primary_shape, calculate_layer_dimensions, self_multiply

# %%
# Test extract_primary_shape with single output tuple
# Expected: returns the same tuple

def test_extract_primary_shape_single_tuple():
    shape = (None, 224, 224, 3)
    result = extract_primary_shape(shape, layer_name='test')
    assert result == (None, 224, 224, 3)

# %%
# Test extract_primary_shape with multi-output tuple
# Expected: warns about multi-output and returns first shape

def test_extract_primary_shape_multi_tuple_warn():
    shape = ((None, 197, 1024), (None, 16, None, None))
    with pytest.warns(UserWarning) as record:
        result = extract_primary_shape(shape, layer_name='multi')
    assert result == (None, 197, 1024)
    # Warning mentions "Multi-output layer detected"
    assert any('Multi-output layer detected' in str(w.message) for w in record)

# %%
# Test extract_primary_shape with single-element list
# Expected: returns the list's only element

def test_extract_primary_shape_list_single():
    shape = [(None, 10, 5)]
    result = extract_primary_shape(shape, layer_name='list_single')
    assert result == (None, 10, 5)

# %%
# Test extract_primary_shape with multi-element list
# Expected: warns and returns the first element

def test_extract_primary_shape_list_multiple_warn():
    shape = [(None, 8), (None, 4)]
    with pytest.warns(UserWarning):
        result = extract_primary_shape(shape, layer_name='list_multi')
    assert result == (None, 8)

# %%
# Test extract_primary_shape with empty list
# Expected: warns and returns default (None,1)

def test_extract_primary_shape_empty_list_warn():
    shape = []
    with pytest.warns(UserWarning):
        result = extract_primary_shape(shape, layer_name='empty')
    assert result == (None, 1)

# %%
# Test extract_primary_shape with None
# Expected: warns and returns default (None,1)

def test_extract_primary_shape_none_warn():
    shape = None
    with pytest.warns(UserWarning):
        result = extract_primary_shape(shape, layer_name='none')
    assert result == (None, 1)

# %%
# Test extract_primary_shape with unsupported type
# Expected: raises RuntimeError

def test_extract_primary_shape_unsupported_type_error():
    shape = 123
    with pytest.raises(RuntimeError):
        extract_primary_shape(shape, layer_name='bad')

# %%
# Test self_multiply utility function
# Expected: multiplies tuple values

def test_self_multiply_tuple():
    tup = (2, 3, 4)
    result = self_multiply(tup)
    assert result == 2 * 3 * 4

# %%
# Test calculate_layer_dimensions in accurate mode
# Expected: follows original block behavior

def test_calculate_dimensions_accurate_1d_y():
    shape = (None, 10)
    x, y, z = calculate_layer_dimensions(shape, scale_z=0.5, scale_xy=1, max_z=100, max_xy=100,
                                         min_z=1, min_xy=1, one_dim_orientation='y', sizing_mode='accurate')
    # For 1D y orientation: y = clamp(10*1,1,100)=10; x=min_xy; z=min_z
    assert (x, y, z) == (1, 10, 1)

# %%
# Test calculate_layer_dimensions in accurate mode z-orientation
# Expected: z computed for 1D

def test_calculate_dimensions_accurate_1d_z():
    shape = (None, 20)
    x, y, z = calculate_layer_dimensions(shape, scale_z=0.5, scale_xy=1, max_z=5, max_xy=100,
                                         min_z=2, min_xy=1, one_dim_orientation='z', sizing_mode='accurate')
    # z = clamp(20*0.5,2,5)=clamp(10,2,5)=5; x,y=min
    assert (x, y, z) == (1, 1, 5)

# %%
# Test calculate_layer_dimensions in accurate mode 2D
# Expected: dims length==2

def test_calculate_dimensions_accurate_2d():
    shape = (None, 4, 5)
    x, y, z = calculate_layer_dimensions(shape, scale_z=2, scale_xy=3, max_z=30, max_xy=30,
                                         min_z=1, min_xy=1, one_dim_orientation='y', sizing_mode='accurate')
    # x=clamp(4*3)=12; y=clamp(5*3)=15; z=clamp(5*2)=10
    assert (x, y, z) == (12, 15, 10)

# %%
# Test calculate_layer_dimensions in accurate mode >=3D
# Expected: dims length>=3 uses self_multiply

def test_calculate_dimensions_accurate_3d():
    shape = (None, 2, 3, 4)
    x, y, z = calculate_layer_dimensions(shape, scale_z=1, scale_xy=1, max_z=100, max_xy=100,
                                         min_z=1, min_xy=1, one_dim_orientation='y', sizing_mode='accurate')
    # z=self_multiply([4])=4
    assert (x, y, z) == (2, 3, 4)

# %%
# Test calculate_layer_dimensions in capped mode
# Expected: dims capped at provided caps

def test_calculate_dimensions_capped():
    shape = (None, 10, 20)
    x, y, z = calculate_layer_dimensions(shape, scale_z=10, scale_xy=10, max_z=999, max_xy=999,
                                         min_z=1, min_xy=1, one_dim_orientation='y',
                                         sizing_mode='capped', dimension_caps={'sequence':15,'channels':50})
    # x=clamp(10*10,1,15)=15; y=clamp(20*10,1,15)=15; z=clamp(shape[2]*scale_z,1,50)=clamp(20*10,1,50)=50
    assert (x, y, z) == (15, 15, 50)

# %%
# Test calculate_layer_dimensions in logarithmic mode small dimension
# Expected: similar to min

def test_calculate_dimensions_logarithmic_small():
    shape = (None, 1)
    x, y, z = calculate_layer_dimensions(shape, scale_z=1, scale_xy=1, max_z=10, max_xy=10,
                                         min_z=2, min_xy=2, one_dim_orientation='y', sizing_mode='logarithmic')
    # y log scaling: value<=1 -> min_xy
    assert (x, y, z) == (2, 2, 2)

# %%
# Test calculate_layer_dimensions in balanced mode for large dims
# Expected: balanced scaling less than accurate

def test_calculate_dimensions_balanced_large():
    shape = (None, 1024, 512)
    x_acc, y_acc, z_acc = calculate_layer_dimensions(shape, scale_z=1, scale_xy=1, max_z=2000, max_xy=2000,
                                                   min_z=1, min_xy=1, one_dim_orientation='y', sizing_mode='accurate')
    x_bal, y_bal, z_bal = calculate_layer_dimensions(shape, scale_z=1, scale_xy=1, max_z=2000, max_xy=2000,
                                                   min_z=1, min_xy=1, one_dim_orientation='y', sizing_mode='balanced')
    # balanced should reduce dimensions
    assert x_bal < x_acc and y_bal < y_acc
