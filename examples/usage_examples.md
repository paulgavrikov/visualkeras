# Usage Examples for VisualKeras

## Layered View

### Basic Usage
```python
import visualkeras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

# Define a simple sequential model
simple_sequential_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Basic usage of visualkeras
basic_layered_img = visualkeras.layered_view(simple_sequential_model, draw_funnel=False)

# Display the image
basic_layered_img.show()
```

![An example of layered style visualization on a simple sequential model with little styling](https://raw.githubusercontent.com/paulgavrikov/visualkeras/refs/heads/master/figures/figure1_layered_basic.png)

### Advanced Usage
```python
import visualkeras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from PIL import ImageFont
from collections import defaultdict

# Define custom font for the model visualization
custom_font = ImageFont.truetype("arial.ttf", 24)

# Define custom color map
color_map = defaultdict(dict)
color_map[Conv2D]['fill'] = 'orange'
color_map[MaxPooling2D]['fill'] = 'red'
color_map[Dense]['fill'] = 'teal'

# Define a larger sequential model with more complexity
complex_sequential_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Advanced usage of visualkeras with custom styling
advanced_layered_img = visualkeras.layered_view(
    complex_sequential_model,
    legend=True,                    # Show legend
    font=custom_font,               # Custom font for legend
    color_map=color_map,            # Custom colors
    draw_volume=True,               # 3D volumetric rendering
    draw_funnel=True,               # Show funnel connectors
    spacing=50,                     # Increase spacing between layers
    padding=30,                     # Add padding around the visualization
    scale_xy=2,                     # Scale x-y dimensions
    scale_z=1,                      # Scale z dimension
    max_z=400,                      # Cap maximum z dimension
    font_color='black',             # Legend font color
    one_dim_orientation='y',        # Orientation for 1D layers
    sizing_mode='accurate',         # Use balanced sizing for layers
    type_ignore=[Flatten, Dropout], # Ignore Flatten and Dropout layers
)

# Display the image
advanced_layered_img.show()
```

![An example of a more complex model's Layered View with custom styling](https://raw.githubusercontent.com/paulgavrikov/visualkeras/refs/heads/master/figures/figure2_layered_advanced.png)

## Graph View

### Basic Usage
```python
import visualkeras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

# Define a simple sequential model
simple_sequential_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Basic usage of visualkeras to create a graph view
basic_graph_img = visualkeras.graph_view(simple_sequential_model)

# Display the image
basic_graph_img.show()
```

![An example of Graph View visualization on a simple sequential model with little styling](https://raw.githubusercontent.com/paulgavrikov/visualkeras/refs/heads/master/figures/figure3_graph_basic.png)

### Advanced Usage
```python
import visualkeras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from collections import defaultdict

# Define a larger sequential model with more complexity
complex_sequential_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Define custom color map for the graph view
graph_color_map = defaultdict(dict)
graph_color_map[Conv2D]['fill'] = 'lightblue'
graph_color_map[Conv2D]['outline'] = 'darkblue'
graph_color_map[MaxPooling2D]['fill'] = 'lightcoral'
graph_color_map[MaxPooling2D]['outline'] = 'darkred'
graph_color_map[Flatten]['fill'] = 'lightgreen'
graph_color_map[Flatten]['outline'] = 'darkgreen'
graph_color_map[Dense]['fill'] = 'lightyellow'
graph_color_map[Dense]['outline'] = 'darkorange'
graph_color_map[Dropout]['fill'] = 'lightpink'
graph_color_map[Dropout]['outline'] = 'purple'

# Create advanced graph view with customizations
advanced_graph_img = visualkeras.graph_view(
    complex_sequential_model,
    color_map=graph_color_map,     # Custom color scheme
    node_size=60,                  # Larger nodes
    connector_fill='gray',         # Gray connectors
    connector_width=2,             # Thicker connectors
    layer_spacing=180,             # More spacing between layers
    node_spacing=40,               # Spacing between nodes in same layer
    padding=40,                    # Padding around the diagram
    background_fill='white',       # White background
    ellipsize_after=8              # Ellipsize layers with >8 neurons
)

# Display the image
advanced_graph_img.show()
```

![An example of a more complex model's Graph View with custom styling](https://raw.githubusercontent.com/paulgavrikov/visualkeras/refs/heads/master/figures/figure4_graph_advanced.png)