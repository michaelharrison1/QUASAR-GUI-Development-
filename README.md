# QUASAR EEG/ECG Multichannel Viewer

An interactive plotting application for visualizing EEG and ECG time-series data with advanced navigation and channel management capabilities.

## Features

- Separate subplots for EEG (µV), ECG (mV), and reference channels (CM)
- Built-in range slider, pan, zoom, and time-based navigation buttons
- Quick selection of EEG channels by anatomical regions (Frontal, Central, Parietal, etc.)
- Toggle visibility of individual channels through interactive legend
- Data decimation for large datasets while preserving signal integrity

### Dependencies
Install the required Python packages using:

```bash
pip install pandas plotly numpy
```

### System Requirements
- Python 3.7+
- Modern web browser for viewing HTML output
- Minimum 4GB RAM for large datasets

## How to Run

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd QUASAR-GUI-Development-
   ```

2. **Ensure your CSV file is present**:
   - Place `EEG and ECG data_02_raw.csv` in the project directory
   - The script expects this exact filename

3. **Run the plotting script**:
   ```bash
   python plot_eeg_improved.py
   ```

4. **View the output**:
   - Open `eeg_ecg_plot_improved.html` in your web browser
   - The plot will load with interactive controls

## Usage Guide

### Navigation
- **Range Slider**: Use the bottom slider for quick time navigation
- **Mouse Controls**: 
  - Drag to pan
  - Scroll wheel to zoom
  - Double-click to reset zoom
- **Time Buttons**: Quick jump to first 10s, last 10s, or full range

### Channel Management
- **Legend**: Click channel names to toggle visibility
- **Brain Region Buttons**: Show/hide channels by anatomical regions
- **Show All/Hide All**: Bulk channel visibility controls

## Design Choices

### Scaling Strategy
The application addresses the significant amplitude differences between signal types through a **multi-subplot approach**:

1. **EEG Channels (Top subplot)**: 
   - Displayed in microvolts (µV) with 200µV vertical offsets between channels
   - Optimized for typical EEG amplitude ranges (10-200µV)

2. **ECG Channels (Middle subplot)**:
   - Converted from µV to millivolts (mV) by dividing by 1000
   - Handles typical ECG amplitudes (~mV range)

3. **Reference Channel (Bottom subplot)**:
   - CM (Common Mode) reference displayed separately in mV
   - Prevents interference with physiological signal interpretation

### Performance Optimizations
- **Data Decimation**: Large datasets (>50,000 points) are automatically decimated by default factor of 3
- **Memory Efficiency**: Uses float32 instead of float64 to reduce memory usage
- **Selective Loading**: Only processes relevant channels, ignoring metadata columns
- **Efficient Rendering**: Initial display shows only first 5 EEG channels to improve load times

### User Experience Features
- **Brain Region Organization**: EEG channels grouped by anatomical location for clinical relevance
- **Unified Hover**: Cross-subplot cursor synchronization for temporal alignment
- **Professional Styling**: Medical-grade color scheme and typography
- **Responsive Legend**: Positioned to avoid overlap with plot area

## AI Assistance

I consulted AI to explore performance optimization techniques. The AI guided me through various approaches including data decimation strategies, memory optimization with float32 data types, and efficient rendering techniques. Additionally, I used AI to help design the brain region grouping system, asking for guidance on how to categorize EEG channels by anatomical locations (Frontal, Central, Parietal, Temporal, Occipital, and Auricular regions) and implement the interactive button controls that allow users to quickly focus on specific brain areas.

## Future Work

In the future, these features would be added. Due to time constraints they were not added in this version of the plot. 

- **Annotation System**: Allow users to mark events and add comments directly on the plot
- **Data Export**: Export filtered data or selected time segments to CSV
- **Multi-file Support**: Load and compare multiple recording sessions
- **Custom Scaling**: User-defined amplitude scaling and offset controls
- **Artifact Detection**: Automatic identification of signal artifacts


## Project Structure

```
QUASAR-GUI-Development-/
├── plot_eeg.py          # Main plotting application
├── eeg_ecg_plot.html    # Generated interactive plot
├── EEG and ECG data_02_raw.csv   # Input data file
├── Requirements.txt              # Project requirements document
└── README.md                     # This file
```