========================================================
Tutorial 5: Debugging Visualizations with Parameters
========================================================

*Estimated time: 30 minutes*

Learn how to fix common visualization issues by adjusting visualkeras parameters. This tutorial shows practical solutions to layout problems, crowding, text overlap, and visual confusion. These are common problems when you are working with custom visualization options in visualkeras.

Overview
========

In this tutorial, you'll learn how to:

- Diagnose what's wrong with a visualization
- Adjust spacing and padding to fix crowding
- Use scaling parameters to control layer sizing
- Fix text overlap and label visibility
- Optimize for specific model sizes and complexities

Problem: Layers Too Crowded
===========================

**The Issue:** When the visualization is cramped or hard to read, layers overlap or blend together.

.. code-block:: python

    import visualkeras
    from tensorflow import keras

    # Build a typical CNN
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10),
    ])

    # This may look crowded:
    image = visualkeras.show(model, mode='layered')
    image.show()

**The Solution:** Increase spacing and padding

.. code-block:: python

    # Solution 1: Use preset='compact' for automatic optimization
    image = visualkeras.show(model, mode='layered', preset='compact')

    # Solution 2: Manually increase spacing between layers
    image = visualkeras.show(
        model,
        mode='layered',
        spacing=20,  # Default: 10
        padding=15,  # Default: 10
    )

    # Solution 3: Combine spacing with reduced scaling
    image = visualkeras.show(
        model,
        mode='layered',
        spacing=15,
        padding=15,
        scale_xy=3,  # Default: 4 - reduce layer width slightly
        scale_z=1.0,  # Default: 1.5 - reduce 3D depth
    )

**Fine-tuning for different model sizes:**

.. code-block:: python

    def visualize_model(model, model_size='medium'):
        """Auto-adjust spacing based on model size."""
        
        if model_size == 'small':  # <20 layers
            return visualkeras.show(model, mode='layered', spacing=8, padding=8)
        
        elif model_size == 'medium':  # 20-50 layers
            return visualkeras.show(model, mode='layered', spacing=12, padding=10)
        
        elif model_size == 'large':  # 50+ layers
            return visualkeras.show(
                model,
                mode='layered',
                spacing=10,
                padding=5,
                scale_xy=2,
                scale_z=0.8,
            )

    # Detect and apply automatically
    num_layers = len(model.layers)
    if num_layers < 20:
        size = 'small'
    elif num_layers < 50:
        size = 'medium'
    else:
        size = 'large'

    image = visualize_model(model, size)
    image.show()

Problem: Layers Too Small
=========================

**The Issue:** With large models, individual layers become tiny and details are lost.

.. code-block:: python

    # This might produce tiny layers:
    large_model = keras.Sequential([
        keras.layers.Input(shape=(512, 512, 3)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        # ... 50 more layers ...
    ])

    image = visualkeras.show(large_model, mode='layered')
    image.show()

**The Solutions:**

.. code-block:: python

    # Solution 1: Increase scale_xy and scale_z to make layers bigger
    image = visualkeras.show(
        large_model,
        mode='layered',
        scale_xy=6,  # Default: 4 - make layers wider
        scale_z=2.5,  # Default: 1.5 - make layers deeper
    )

    # Solution 2: Reduce padding/spacing to fit more screen space
    image = visualkeras.show(
        large_model,
        mode='layered',
        spacing=6,
        padding=5,
        scale_xy=5,
    )

    # Solution 3: Override min/max sizing constraints
    image = visualkeras.show(
        large_model,
        mode='layered',
        min_xy=25,  # Default: 20 - enforce minimum width
        min_z=30,   # Default: 20 - enforce minimum depth
        scale_xy=4,
        scale_z=2,
    )

    # Solution 4: Use 'full_range' sizing mode
    # This stretches dimensions to use full min-max range
    image = visualkeras.show(
        large_model,
        mode='layered',
        sizing_mode='full_range',  # vs default 'accurate'
        scale_xy=4,
    )

Problem: Text Overlap and Labels
================================

**The Issue:** Layer names and dimensions overlap or are hard to read.

.. code-block:: python

    # Problem: text overlapping
    image = visualkeras.show(model, mode='layered', text_callable=None)
    image.show()

**The Solutions:**

.. code-block:: python

    # Solution 1: Increase vertical spacing for text
    image = visualkeras.show(
        model,
        mode='layered',
        text_vspacing=8,  # Default: 4 - more vertical space
    )

    # Solution 2: Remove text labels entirely for compact view
    image = visualkeras.show(
        model,
        mode='layered',
        text_callable=None,  # No text
    )

    # Solution 3: Show shape information above layers instead
    from visualkeras.options import LAYERED_TEXT_CALLABLES
    
    image = visualkeras.show(
        model,
        mode='layered',
        text_callable=LAYERED_TEXT_CALLABLES['shape'],  # Just dimensions
    )

    # Solution 4: Increase spacing AND text spacing
    image = visualkeras.show(
        model,
        mode='layered',
        spacing=15,
        text_vspacing=6,
        text_callable=LAYERED_TEXT_CALLABLES['name_shape'],  # Compact label
    )

    # Solution 5: Use custom text callable to shorten names
    def short_names(index, layer):
        """Custom: show layer type only, not full name."""
        layer_type = type(layer).__name__
        return (layer_type, False)  # (text, above)

    image = visualkeras.show(
        model,
        mode='layered',
        text_callable=short_names,
    )

Problem: Layers Beyond Output Bounds
====================================

**The Issue:** The visualization extends beyond screen/page bounds or has weird proportions.

.. code-block:: python

    # Problem: image is too wide or too tall
    image = visualkeras.show(model, mode='layered')
    print(f"Image size: {image.size}")  # (5400, 300) - too wide!

**The Solutions:**

.. code-block:: python

    # Solution 1: Reduce scale_xy to make diagram narrower
    image = visualkeras.show(
        model,
        mode='layered',
        scale_xy=2,  # Make layers skinnier (default: 4)
    )

    # Solution 2: Enable show_dimension to understand size claims
    image = visualkeras.show(
        model,
        mode='layered',
        show_dimension=True,  # Debug: show actual layer dims
        scale_xy=3,
    )

    # Solution 3: Reduce max_xy limits
    image = visualkeras.show(
        model,
        mode='layered',
        max_xy=1000,  # Default: 2000 - cap maximum layer width
        scale_xy=3,
    )

    # Solution 4: Adjust for target image size
    # If you want output ~800px wide:
    image = visualkeras.show(
        model,
        mode='layered',
        spacing=8,
        padding=10,
        scale_xy=2.5,
    )
    
    width, height = image.size
    if width > 800:
        # Resize down while keeping quality
        scale = 800 / width
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

Problem: Layer Details Lost
===========================

**The Issue:** When showing very wide layers (e.g., high-resolution feature maps), critical details aren't visible.

.. code-block:: python

    # Wide layers (large channel counts or feature maps)
    model = keras.Sequential([
        keras.layers.Input(shape=(256, 256, 3)),
        keras.layers.Conv2D(256, (3, 3), activation='relu'),  # Many channels
        keras.layers.Conv2D(512, (3, 3), activation='relu'),  # Even more
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
    ])

    # Problem: layers are too wide to see clearly
    image = visualkeras.show(model, mode='layered')
    image.show()

**The Solutions:**

.. code-block:: python

    # Solution 1: Cap maximum layer width
    image = visualkeras.show(
        model,
        mode='layered',
        max_xy=500,  # Don't let individual layers exceed 500px
        scale_xy=3,
    )

    # Solution 2: Switch to graph view for complex shapes
    image = visualkeras.show(model, mode='graph')

    # Solution 3: Use sizing_mode='full_range' to distribute space
    image = visualkeras.show(
        model,
        mode='layered',
        sizing_mode='full_range',
        max_xy=300,
    )

    # Solution 4: Show dimension information to understand scaling
    image = visualkeras.show(
        model,
        mode='layered',
        show_dimension=True,  # Display actual dimensions as text
        max_xy=400,
    )

Problem: Wrong Visualization Mode
=================================

**The Issue:** The model looks confusing or unclear in the chosen visualization mode.

**Decision tree to pick the right mode:**

.. code-block:: python

    def choose_best_mode(model):
        """Suggest the best visualization mode for a model."""
        
        # Check model complexity
        num_layers = len(model.layers)
        has_skip_connections = hasattr(model, 'optimizer')  # Simplistic check
        
        try:
            # Sequential models are simple
            model.layers
            model.build((None,) + model.input_shape[1:])
            
            if num_layers < 15:
                return 'layered'  # Simple: use layered
            elif num_layers < 50:
                return 'graph'    # Medium: use graph
            else:
                return 'functional'  # Complex: use functional
        
        except:
            # Functional model with complex connections
            return 'functional'

    # Use it
    mode = choose_best_mode(model)
    image = visualkeras.show(model, mode=mode)
    image.show()

    # Or visualize all four modes to compare
    print("Finding best visualization mode...")
    for mode in ['layered', 'graph', 'functional', 'lenet']:
        try:
            image = visualkeras.show(model, mode=mode)
            width, height = image.size
            print(f"{mode:12} -> {width:5}x{height:5} pixels")
            image.save(f'mode_comparison_{mode}.png')
        except Exception as e:
            print(f"{mode:12} -> ERROR: {e}")

    print("Compare the saved images to choose the best mode")

Parameter Adjustment Workflow
=============================

**Systematic approach to debugging visualizations:**

.. code-block:: python

    def debug_visualization(model, mode='layered', max_iterations=5):
        """Iteratively refine visualization parameters."""
        
        params = {
            'spacing': 10,
            'padding': 10,
            'scale_xy': 4,
            'scale_z': 1.5,
            'text_vspacing': 4,
        }
        
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            
            try:
                image = visualkeras.show(model, mode=mode, **params)
                width, height = image.size
                
                print(f"\nIteration {iteration}:")
                print(f"  Size: {width}x{height} pixels")
                print(f"  Params: {params}")
                
                # Assess output
                if width > 1500:
                    print("  → Too wide, reducing scale_xy")
                    params['scale_xy'] *= 0.8
                elif width < 400:
                    print("  → Too narrow, increasing scale_xy")
                    params['scale_xy'] *= 1.2
                elif height > 1500:
                    print("  → Too tall, reducing spacing")
                    params['spacing'] = max(5, params['spacing'] - 2)
                else:
                    print("  ✓ Size looks good!")
                    image.save(f'debug_iteration_{iteration}.png')
                    return image, params
            
            except Exception as e:
                print(f"  ERROR: {e}")
                break
        
        return None, params

    # Use it
    best_image, best_params = debug_visualization(model)
    print(f"\nBest parameters: {best_params}")

Preset Comparison
==================

Understanding when to use each preset:

.. code-block:: python

    from visualkeras.options import LAYERED_PRESETS

    # Compare presets visually
    presets = ['default', 'compact', 'presentation']
    
    for preset_name in presets:
        image = visualkeras.show(model, mode='layered', preset=preset_name)
        width, height = image.size
        print(f"{preset_name:15} -> {width:5}x{height:5} -> {width*height:8} pixels")
        image.save(f'preset_{preset_name}.png')

    # Output to compare:
    # default        ->  1200x  400 ->  480000 pixels
    # compact        ->  1050x  350 ->  367500 pixels (30% smaller)
    # presentation   ->  1500x  450 ->  675000 pixels (40% larger)

    # Use for different contexts:
    # - 'compact': Tight layouts, screen display, documentation
    # - 'default': General purpose, balanced view
    # - 'presentation': Papers, slides, high-impact visuals

Performance Tips
================

When debugging multiple parameter combinations:

.. code-block:: python

    import time

    def time_visualization(model, params):
        """Measure how long a visualization takes."""
        start = time.time()
        image = visualkeras.show(model, mode='layered', **params)
        duration = time.time() - start
        width, height = image.size
        return image, duration, (width, height)

    # Test different parameter sets
    test_params = [
        {'scale_xy': 2, 'scale_z': 1},
        {'scale_xy': 4, 'scale_z': 1.5},
        {'scale_xy': 6, 'scale_z': 2},
    ]

    for params in test_params:
        image, duration, size = time_visualization(model, params)
        print(f"Params {params} -> {duration:.2f}s, {size} pixels")

Summary
========

You've learned how to:

- Debug crowded visualizations with spacing and padding
- Fix small layer sizes with scaling parameters
- Resolve text overlap with text spacing and custom callables
- Handle oversized outputs with max constraints
- Choose the right visualization mode for different models
- Systematically adjust parameters to achieve desired results
- Compare presets and understand when to use each

Key parameters for debugging:

- ``spacing``: Gap between layers (default: 10)
- ``padding``: Border around visualization (default: 10)
- ``scale_xy``: Width/height multiplier (default: 4)
- ``scale_z``: Depth multiplier (default: 1.5)
- ``text_vspacing``: Vertical space for text (default: 4)
- ``min_xy``, ``max_xy``: Layer size constraints
- ``sizing_mode``: 'accurate' or 'full_range'
- ``text_callable``: Control what text appears