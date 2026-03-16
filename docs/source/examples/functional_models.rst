==================
Functional Models
==================

Visualizing complex models with multiple paths, branches, and skip connections.

Understanding the Functional Visualization Mode
================================================

The ``functional_view()`` mode is designed specifically for Keras Functional models. It combines the clarity of layered visualization (showing layer stacking) with graph awareness (showing connections). This makes it ideal for:

- **Multi-branch architectures** (multiple input/output paths)
- **Skip connections and residual networks**
- **Models with repeated layer sequences** (can be auto-collapsed)
- **Citation/publication diagrams** that need structure and detail

Compare with other modes:

- **Graph mode**: Shows computational structure clearly but can be dense and hard to parse visually
- **Functional mode**: Maintains layer stacking for readability while showing connections
- **Layered mode**: Best for simple sequential models, doesn't handle complex branching well
- **LeNet mode**: Best for detailed channel-by-channel visualization

Multi-Input / Multi-Output Model
=================================

A model that fuses data from two input branches:

.. code-block:: python

    import tensorflow as tf
    from tensorflow import keras
    import visualkeras

    # Define inputs
    input_image = keras.Input(shape=(32, 32, 3), name='image')
    input_features = keras.Input(shape=(10,), name='features')

    # Image processing branch
    x1 = keras.layers.Conv2D(32, (3, 3), activation='relu')(input_image)
    x1 = keras.layers.MaxPooling2D((2, 2))(x1)
    x1 = keras.layers.Conv2D(64, (3, 3), activation='relu')(x1)
    x1 = keras.layers.Flatten()(x1)

    # Feature processing branch
    x2 = keras.layers.Dense(64, activation='relu')(input_features)

    # Merge branches
    merged = keras.layers.Concatenate()([x1, x2])
    outputs = keras.layers.Dense(10, activation='softmax')(merged)

    model = keras.Model(inputs=[input_image, input_features], outputs=outputs)
    
    # Visualize with functional mode (shows branches clearly while maintaining layer stacking)
    image = visualkeras.show(model, mode='functional')
    image.show()

**Why functional mode here?** It clearly shows how the two input branches converge through concatenation, while maintaining visual layer structure. Graph mode would also work but can be harder to parse visually.

Residual Network Block
======================

A simple residual connection where a layer's output is added back to its input:

.. code-block:: python

    inputs = keras.Input(shape=(224, 224, 3))

    # Initial convolution
    x = keras.layers.Conv2D(64, (7, 7), padding='same', activation='relu')(inputs)
    x = keras.layers.BatchNormalization()(x)

    # Residual block: capture input for skip connection
    residual = x
    x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    # Skip connection: add input back
    x = keras.layers.Add()([x, residual])
    x = keras.layers.ReLU()(x)

    # Rest of model
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(1000, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Visualize with functional mode (clearly shows the Add skip connection)
    image = visualkeras.show(model, mode='functional', preset='presentation')
    image.show()

**Why functional mode?** It reveals the skip connection (Add layer) that would be invisible or confusing in layered mode.

Comparing Visualization Modes
==============================

Here's the same multi-input model visualized in all modes:

.. code-block:: python

    import visualkeras
    
    # Create the multi-branch model (same as above)
    # ... model definition ...
    
    # Compare all visualization modes
    
    # Functional: Best for understanding flow with layer structure
    functional = visualkeras.show(model, mode='functional')
    
    # Graph: Shows structure clearly, can be dense
    graph = visualkeras.show(model, mode='graph')
    
    # Layered: Works better for simpler models
    layered = visualkeras.show(model, mode='layered')
    
    # LeNet: Good for detailed channel-by-channel visualization
    lenet = visualkeras.show(model, mode='lenet')

**Recommendation**: Use ``functional`` mode as your default for Keras Functional models. Switch to ``graph`` mode if you need the most explicit connectivity diagram.

Multi-Output Model with Different Tasks
========================================

A model performing both classification and regression:

.. code-block:: python

    inputs = keras.Input(shape=(224, 224, 3))

    # Shared feature extraction
    x = keras.layers.Conv2D(64, (7, 7), padding='same', activation='relu')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation='relu')(x)

    # Classification branch
    classification = keras.layers.Dense(10, activation='softmax', name='classification')(x)

    # Regression branch  
    regression = keras.layers.Dense(1, name='regression')(x)

    model = keras.Model(inputs=inputs, outputs=[classification, regression])
    
    # Functional mode shows both output branches clearly
    image = visualkeras.show(model, mode='functional', preset='presentation')
    image.show()

Complex Multi-Branch Inception-like Module
============================================

A model with multiple parallel branches at different filter sizes:

.. code-block:: python

    inputs = keras.Input(shape=(224, 224, 3))

    # Branch 1: 1x1 convolution
    branch1x1 = keras.layers.Conv2D(64, (1, 1), activation='relu')(inputs)

    # Branch 2: 1x1 -> 3x3
    branch3x3 = keras.layers.Conv2D(96, (1, 1), activation='relu')(inputs)
    branch3x3 = keras.layers.Conv2D(128, (3, 3), activation='relu')(branch3x3)

    # Branch 3: 1x1 -> 5x5
    branch5x5 = keras.layers.Conv2D(16, (1, 1), activation='relu')(inputs)
    branch5x5 = keras.layers.Conv2D(32, (5, 5), activation='relu')(branch5x5)

    # Branch 4: Max pooling -> 1x1
    branch_pool = keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = keras.layers.Conv2D(32, (1, 1), activation='relu')(branch_pool)

    # Concatenate all branches
    output = keras.layers.Concatenate()(
        [branch1x1, branch3x3, branch5x5, branch_pool]
    )
    output = keras.layers.MaxPooling2D((2, 2))(output)
    output = keras.layers.Flatten()(output)
    output = keras.layers.Dense(10, activation='softmax')(output)

    model = keras.Model(inputs=inputs, outputs=output)
    
    # Functional mode excels at showing parallel branches
    image = visualkeras.show(model, mode='functional', preset='presentation')
    image.show()

Functional Mode Options
=======================

Fine-tune your functional visualization with mode-specific options:

.. code-block:: python

    from visualkeras.options import FunctionalOptions, FUNCTIONAL_PRESETS

    # Use presentation preset and customize
    options = FunctionalOptions(
        **FUNCTIONAL_PRESETS['presentation'].__dict__,
        column_spacing=100,       # Space between columns
        row_spacing=60,           # Space between rows
        sizing_mode='balanced',   # Better size distribution
        connector_width=2,        # Thicker connection lines
    )

    image = visualkeras.show(model, mode='functional', options=options)
    image.show()

When to Use ``mode='functional'``
=================================

Choose ``functional`` mode when:

✅ **DO** use functional mode for:
    - Keras Functional API models
    - Models with skip connections or residual blocks
    - Multi-input or multi-output architectures
    - Models where showing layer stacking matters
    - Publication diagrams where structure must be clear

❌ **DON'T** use functional mode for:
    - Very deep models (>100 layers) → try graph or layered with collapse
    - Simple sequential models → easier with layered
    - Very wide models with many parallel branches → graph mode may be clearer
    - Detailed channel visualization → use lenet mode

See Also
========

- :doc:`cnn_models` for convolutional network examples
- :doc:`sequential_models` for simple models
- :doc:`../tutorials/index` for detailed tutorials
- :doc:`../api/functional` for functional_view() API reference
