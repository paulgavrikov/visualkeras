======================================================
Tutorial 4: Export, Save & Integration Workflows
======================================================

*Estimated time: 25 minutes*

Learn how to save visualizations in different formats, integrate them into ML workflows, and batch process model visualizations.

Overview
========

In this tutorial, you'll learn how to:

- Save visualizations in multiple file formats (PNG, SVG, PDF)
- Optimize output quality and file size
- Integrate visualizations into training pipelines
- Batch process multiple models
- Reproduce visualizations reliably

File Format Options
===================

visualkeras generates PIL Image objects that can be saved in any format PIL supports.

**PNG — Best for web and general use**

.. code-block:: python

    import visualkeras
    from tensorflow import keras

    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='softmax')
    ])

    # PNG with default quality
    image = visualkeras.show(model, mode='layered')
    image.save('model_architecture.png')

    # For higher quality PNG with better compression
    image.save('model_architecture_hq.png', quality=95, optimize=True)

**SVG — Best for scalable, editable diagrams**

SVG (Scalable Vector Graphics) output is useful when you need to edit the visualization later or want unlimited scaling without quality loss:

.. code-block:: python

    # Convert PIL Image to SVG using cairosvg (install: pip install cairosvg)
    import io
    from PIL import Image

    image = visualkeras.show(model, mode='layered')
    
    # Save as PNG first, then convert to SVG
    png_buffer = io.BytesIO()
    image.save(png_buffer, format='PNG')
    png_buffer.seek(0)

    # Or use external tools: convert model.png model.svg (ImageMagick)
    image.save('model_architecture.png')
    # Then: $ convert model_architecture.png model_architecture.svg


**PDF — Best for documents and printing**

.. code-block:: python

    # PDF output for papers and documents
    image = visualkeras.show(model, mode='layered', preset='presentation')
    image.save('model_architecture.pdf')

    # For print-ready PDF with high DPI
    # PIL doesn't support DPI in PDF directly, but you can resize
    width, height = image.size
    # Create high-DPI version (300 DPI)
    dpi_scale = 2  # 2x for 300 DPI equivalent
    hires_image = image.resize(
        (width * dpi_scale, height * dpi_scale),
        Image.Resampling.LANCZOS
    )
    hires_image.save('model_architecture_print.pdf')

Quality and Size Optimization
==============================

Different use cases need different quality levels:

.. code-block:: python

    import visualkeras

    # 1. Web use — optimize for small file size
    image = visualkeras.show(model, mode='layered')
    image.save('web_small.png', optimize=True)  # ~100-300 KB

    # 2. presentation — balance quality and size
    image = visualkeras.show(model, mode='layered', preset='presentation')
    image.save('presentation.png')  # ~500 KB - 2 MB

    # 3. Publication — maximum quality
    image = visualkeras.show(model, mode='layered', preset='presentation')
    image.save('publication.png', quality=95)  # ~3-5 MB

    # 4. Very large models — use compact preset
    image = visualkeras.show(model, mode='layered', preset='compact')
    image.save('compact.png', optimize=True)  # Smaller & fits on screen

Saving Visualizations During Training
======================================

Automatically save model visualizations at key checkpoints:

.. code-block:: python

    import os
    import visualkeras
    from tensorflow import keras
    import tensorflow as tf

    # Create output directory
    os.makedirs('model_checkpoints', exist_ok=True)

    # Custom callback to visualize model at checkpoints
    class VisualizationCallback(keras.callbacks.Callback):
        def __init__(self, save_dir='model_checkpoints'):
            super().__init__()
            self.save_dir = save_dir

        def on_train_begin(self, logs=None):
            # Save initial architecture
            try:
                image = visualkeras.show(self.model, mode='layered')
                path = os.path.join(self.save_dir, 'initial_architecture.png')
                image.save(path)
                print(f"Saved initial architecture to {path}")
            except Exception as e:
                print(f"Could not visualize model: {e}")

        def on_epoch_end(self, epoch, logs=None):
            # Optionally save every N epochs
            if (epoch + 1) % 10 == 0:
                try:
                    image = visualkeras.show(self.model, mode='graph')
                    path = os.path.join(self.save_dir, f'architecture_epoch_{epoch+1}.png')
                    image.save(path)
                    print(f"Saved epoch {epoch+1} visualization")
                except Exception:
                    pass  # Skip if visualization fails

    # Use the callback
    model = keras.Sequential([...])
    
    callback = VisualizationCallback()
    model.fit(
        x_train, y_train,
        epochs=100,
        callbacks=[callback],
    )

Batch Processing Multiple Models
=================================

Visualize entire model families consistently:

.. code-block:: python

    import visualkeras
    from pathlib import Path

    # Define model configurations
    model_configs = [
        {'name': 'small', 'filters': 32},
        {'name': 'medium', 'filters': 64},
        {'name': 'large', 'filters': 128},
    ]

    # Output directory
    output_dir = Path('model_visualizations')
    output_dir.mkdir(exist_ok=True)

    # Build and visualize each model
    for config in model_configs:
        model = keras.Sequential([
            keras.layers.Conv2D(config['filters'], (3, 3), 
                              activation='relu', 
                              input_shape=(224, 224, 3)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(config['filters']*2, (3, 3), 
                              activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(1000, activation='relu'),
            keras.layers.Dense(10, activation='softmax'),
        ])

        # Use consistent styling across all models
        color_map = {
            keras.layers.Conv2D: {'fill': '#3498db', 'outline': '#2980b9'},
            keras.layers.MaxPooling2D: {'fill': '#2ecc71', 'outline': '#27ae60'},
            keras.layers.Dense: {'fill': '#e74c3c', 'outline': '#c0392b'},
            keras.layers.Flatten: {'fill': '#f39c12', 'outline': '#d68910'},
        }

        # Visualize with consistent options
        image = visualkeras.show(
            model,
            mode='layered',
            preset='presentation',
            color_map=color_map
        )

        # Save with consistent naming
        output_path = output_dir / f'{config["name"]}_architecture.png'
        image.save(output_path, optimize=True)
        print(f"✓ Saved {config['name']} model to {output_path}")

    print(f"\nAll models saved to {output_dir}/")

Integration with Jupyter Notebooks
===================================

Display visualizations directly in notebooks:

.. code-block:: python

    import visualkeras
    from IPython.display import Image as IPImage, display

    model = keras.Sequential([...])

    # Method 1: Direct display (simplest)
    image = visualkeras.show(model, mode='layered')
    display(image)

    # Method 2: Save and display via IPython
    image.save('temp_model.png')
    display(IPImage('temp_model.png'))

    # Method 3: Multiple visualizations side-by-side
    from IPython.display import HTML
    import base64
    import io

    def image_to_base64(pil_image):
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode()

    mode1 = visualkeras.show(model, mode='layered')
    mode2 = visualkeras.show(model, mode='graph')

    html = f"""
    <div style="display: flex; gap: 20px;">
        <div><h3>Layered View</h3><img src="data:image/png;base64,{image_to_base64(mode1)}"></div>
        <div><h3>Graph View</h3><img src="data:image/png;base64,{image_to_base64(mode2)}"></div>
    </div>
    """
    display(HTML(html))

Reproducible Visualization Configurations
==========================================

Store and reuse visualization configurations:

.. code-block:: python

    import json
    import visualkeras
    from visualkeras.options import LayeredOptions

    # Define a reusable configuration
    paper_config = {
        'mode': 'layered',
        'preset': 'presentation',
        'color_map': {
            'Conv2D': {'fill': '#1f77b4', 'outline': '#0d47a1'},
            'MaxPooling2D': {'fill': '#2ca02c', 'outline': '#1b5e20'},
            'Dense': {'fill': '#d62728', 'outline': '#b71c1c'},
        },
        'padding': 20,
        'spacing': 15,
    }

    # Save configuration
    with open('paper_style.json', 'w') as f:
        json.dump(paper_config, f, indent=2)

    # Load and apply configuration
    with open('paper_style.json', 'r') as f:
        config = json.load(f)

    image = visualkeras.show(model, **config)
    image.save('paper_figure.png')

    # Share configurations with team members
    # They can load and use the same style:
    # $ git clone repo
    # $ python apply_style.py

Integration with Documentation Generators
==========================================

Automatically update documentation with current model architecture:

.. code-block:: python

    # docs/generate_archtecture_docs.py
    import visualkeras
    from pathlib import Path
    import sys

    # Add model path to sys.path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Import your models
    from models import create_encoder, create_decoder, create_full_model

    # Generate documentation
    doc_output = Path(__file__).parent / 'generated'
    doc_output.mkdir(exist_ok=True)

    models = {
        'encoder': create_encoder(),
        'decoder': create_decoder(),
        'full_model': create_full_model(),
    }

    for name, model in models.items():
        image = visualkeras.show(model, mode='graph', preset='presentation')
        image.save(doc_output / f'{name}_architecture.png')
        print(f"Generated {name}_architecture.png")

    # Run as: python docs/generate_architecture_docs.py
    # Then commit generated images to docs/generated/

Best Practices
===============

1. **Use presets for consistency**
   - Use ``preset='presentation'`` for papers and documentation
   - Use ``preset='compact'`` for web and tight spaces
   - Define custom presets for team standards

2. **Store configurations in version control**
   - Save color maps and options in JSON
   - Share across team members
   - Track changes to visualization standards

3. **Batch before shipping**
   - Generate all visualizations together
   - Use consistent styling across all images
   - Version images with your model releases

4. **Automate visualization generation**
   - Use callbacks during training
   - Generate in CI/CD pipelines
   - Update documentation automatically

Summary
========

You've learned how to:

- Export visualizations in multiple formats (PNG, SVG, PDF)
- Optimize quality and file size for different use cases
- Integrate visualizations into training workflows
- Batch process multiple models consistently
- Reproduce visualizations with configuration files
- Automate documentation generation

Next, explore how to fix visualization issues and debug your models using visualkeras parameters in :doc:`tutorial_05_debugging_visualizations`.
