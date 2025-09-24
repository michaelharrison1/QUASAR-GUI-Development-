import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
from typing import List, Dict, Tuple

# Config
COLORS = {
    'eeg': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', 
            '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8'],
    'ecg': ['#d62728', '#ff7f0e'],
    'cm': '#9467bd'
}

class EEGPlotter:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self.eeg_channels = []
        self.ecg_channels = []
        self.other_channels = []
        self.time_col = 'Time'
        
    def load_data(self) -> pd.DataFrame:
        """Load and preprocess CSV data with optimizations for speed."""
        print("Loading data...")
        start_time = time.time()
        
        # Find header row more efficiently
        with open(self.file_path, 'r') as f:
            for i, line in enumerate(f):
                if not line.startswith('#'):
                    skiprows = i
                    break
        
        # Load data with optimized parameters
        self.df = pd.read_csv(
            self.file_path, 
            skiprows=skiprows,
            dtype=np.float32,  # Use float32 instead of float64 to save memory
            engine='c'  # Use C engine for faster parsing
        )
        
        # Convert time column to float32 if it's not already
        self.df[self.time_col] = self.df[self.time_col].astype(np.float32)
        
        print(f"Data loaded in {time.time() - start_time:.2f} seconds")
        print(f"Shape: {self.df.shape}")
        return self.df
    
    def identify_channels(self) -> None:
        """Identify and categorize different channel types."""
        all_cols = self.df.columns.tolist()
        
        # Define channel lists based on the data structure
        potential_eeg = [
            'Fz', 'Cz', 'P3', 'C3', 'F3', 'F4', 'C4', 'P4', 'Fp1', 'Fp2',
            'T3', 'T4', 'T5', 'T6', 'O1', 'O2', 'F7', 'F8', 'A1', 'A2', 'Pz'
        ]
        
        potential_ecg = ['X1:LEOG', 'X2:REOG']
        potential_other = ['CM']
        
        # Filter channels that actually exist in the data
        self.eeg_channels = [ch for ch in potential_eeg if ch in all_cols]
        self.ecg_channels = [ch for ch in potential_ecg if ch in all_cols]
        self.other_channels = [ch for ch in potential_other if ch in all_cols]
        
        print(f"Found {len(self.eeg_channels)} EEG channels: {self.eeg_channels}")
        print(f"Found {len(self.ecg_channels)} ECG channels: {self.ecg_channels}")
        print(f"Found {len(self.other_channels)} other channels: {self.other_channels}")
    
    def get_brain_regions(self) -> Dict[str, List[str]]:
        """Categorize EEG channels by brain regions."""
        brain_regions = {
            'Frontal': ['Fz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2'],
            'Central': ['Cz', 'C3', 'C4'],
            'Parietal': ['Pz', 'P3', 'P4'],
            'Temporal': ['T3', 'T4', 'T5', 'T6'],
            'Occipital': ['O1', 'O2'],
            'Auricular': ['A1', 'A2']
        }
        
        # Filter to only include channels that exist in our data
        existing_regions = {}
        for region, channels in brain_regions.items():
            existing_channels = [ch for ch in channels if ch in self.eeg_channels]
            if existing_channels:
                existing_regions[region] = existing_channels
        
        return existing_regions
    
    def decimate_data(self, factor: int = 10) -> pd.DataFrame:
        """Decimate data for faster plotting while preserving key features."""
        if len(self.df) > 50000:  # Only decimate for large datasets
            print(f"Decimating data by factor {factor} for better performance...")
            decimated_df = self.df.iloc[::factor].copy()
            return decimated_df
        return self.df
    
    def create_subplot_structure(self) -> go.Figure:
        """Create optimized subplot structure."""
        # Calculate dynamic row heights
        num_eeg = len(self.eeg_channels)
        num_ecg = len(self.ecg_channels)
        num_other = len(self.other_channels)
        
        # Create subplot with better spacing
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.25, 0.15],
            subplot_titles=('<b>EEG Channels (μV)</b>', '<b>ECG Channels (mV)</b>', '<b>Reference (CM)</b>'),
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        return fig
    
    def add_eeg_traces(self, fig: go.Figure, plot_df: pd.DataFrame) -> None:
        """Add EEG traces with styling."""
        for i, channel in enumerate(self.eeg_channels):
            color = COLORS['eeg'][i % len(COLORS['eeg'])]
            
            # Add offset for better visualization
            offset = i * 200  # 200 μV offset between channels
            y_data = plot_df[channel] + offset
            
            fig.add_trace(
                go.Scatter(
                    x=plot_df[self.time_col],
                    y=y_data,
                    name=f'{channel}',
                    line=dict(width=1, color=color),
                    hovertemplate=f'<b>{channel}</b><br>' +
                                  'Time: %{x:.3f}s<br>' +
                                  'Value: %{customdata:.1f}μV<br>' +
                                  '<extra></extra>',
                    customdata=plot_df[channel],
                    visible='legendonly' if i > 4 else True  # Show only first 5 channels initially
                ),
                row=1, col=1
            )
    
    def add_ecg_traces(self, fig: go.Figure, plot_df: pd.DataFrame) -> None:
        """Add ECG traces with styling."""
        for i, channel in enumerate(self.ecg_channels):
            color = COLORS['ecg'][i % len(COLORS['ecg'])]
            
            fig.add_trace(
                go.Scatter(
                    x=plot_df[self.time_col],
                    y=plot_df[channel] / 1000,  # Convert μV to mV
                    name=f'{channel}',
                    line=dict(width=1.5, color=color),
                    hovertemplate=f'<b>{channel}</b><br>' +
                                  'Time: %{x:.3f}s<br>' +
                                  'Value: %{y:.2f}mV<br>' +
                                  '<extra></extra>',
                    visible=True
                ),
                row=2, col=1
            )
    
    def add_other_traces(self, fig: go.Figure, plot_df: pd.DataFrame) -> None:
        """Add CM and other reference traces."""
        for channel in self.other_channels:
            fig.add_trace(
                go.Scatter(
                    x=plot_df[self.time_col],
                    y=plot_df[channel] / 1000,  # Convert to mV
                    name=f'{channel}',
                    line=dict(width=1, color=COLORS['cm']),
                    hovertemplate=f'<b>{channel}</b><br>' +
                                  'Time: %{x:.3f}s<br>' +
                                  'Value: %{y:.2f}mV<br>' +
                                  '<extra></extra>',
                    visible=True,
                    showlegend=False
                ),
                row=3, col=1
            )
    
    def configure_layout(self, fig: go.Figure) -> None:
        """Configure plot layout with styling."""
        # Get time range
        time_min, time_max = self.df[self.time_col].min(), self.df[self.time_col].max()
        initial_range = min(10, time_max - time_min)  # Show first 10 seconds or full range
        
        fig.update_layout(
            title={
                'text': '<b>EEG/ECG Multichannel Viewer</b>',
                'x': 0.42,
                'xanchor': 'center',
                'font': {'size': 24, 'family': 'Georgia, serif'}
            },
            height=900,
            width=1700,  # Set a fixed width
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.7,
                xanchor="left",
                x=1.05,
                font={'size': 12}
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font={'family': 'Arial, sans-serif'},
            hovermode='x unified',
            margin=dict(r=350)  # Add right margin for legend/buttons
        )
        
        # Configure axes
        fig.update_yaxes(title_text='Amplitude (uv)', row=1, col=1, 
                        gridcolor='lightgray', showgrid=True, zeroline=True)
        fig.update_yaxes(title_text='Amplitude (mV)', row=2, col=1,
                        gridcolor='lightgray', showgrid=True, zeroline=True)
        fig.update_yaxes(title_text='Amplitude (mV)', row=3, col=1,
                        gridcolor='lightgray', showgrid=True, zeroline=True)
        
        # Configure x-axis with range slider
        fig.update_xaxes(
            title_text='Time (seconds)',
            row=3, col=1,
            rangeslider=dict(
                visible=True,
                thickness=0.05
            ),
            range=[time_min, time_min + initial_range],
            gridcolor='lightgray',
            showgrid=True
        )
        
        # Create brain region grouping buttons
        brain_regions = self.get_brain_regions()
        region_buttons = []
        
        # Add general control buttons
        region_buttons.extend([
            dict(
                args=[{"visible": [True] * len(fig.data)}],
                label="Show All",
                method="restyle"
            ),
            dict(
                args=[{"visible": ['legendonly'] * len(fig.data)}],
                label="Hide All",
                method="restyle"
            ),
        ])
        
        # Add brain region buttons
        for region_name, channels in brain_regions.items():
            # Create visibility list: show EEG channels in this region, show ECG/CM, hide other EEG
            visibility_list = []
            for trace in fig.data:
                trace_name = trace.name
                if trace_name in channels:
                    visibility_list.append(True)  # Show this brain region
                elif trace_name in self.ecg_channels or trace_name in self.other_channels:
                    visibility_list.append(True)  # Always show ECG and CM
                elif trace_name in self.eeg_channels:
                    visibility_list.append('legendonly')  # Hide other EEG channels
                else:
                    visibility_list.append(True)  # Show everything else
            
            region_buttons.append(dict(
                args=[{"visible": visibility_list}],
                label=f"{region_name}",
                method="restyle"
            ))
        
        # Add custom buttons for quick navigation and brain regions
        fig.update_layout(
            updatemenus=[
                # Brain region buttons
                dict(
                    type="buttons",
                    direction="down",
                    buttons=region_buttons,
                    pad={"r": 15, "t": 15},
                    showactive=True,
                    x=1.05,
                    xanchor="left",
                    y=1,
                    yanchor="top",
                    font={'size': 12}
                ),
                # Time navigation buttons
                dict(
                    type="buttons",
                    direction="down",
                    buttons=list([
                        dict(
                            args=[{"xaxis.range": [time_min, time_min + 10]}],
                            label="First 10s",
                            method="relayout"
                        ),
                        dict(
                            args=[{"xaxis.range": [time_max - 10, time_max]}],
                            label="Last 10s",
                            method="relayout"
                        ),
                        dict(
                            args=[{"xaxis.range": [time_min, time_max]}],
                            label="Full Range",
                            method="relayout"
                        ),
                    ]),
                    pad={"r": 15, "t": 15},
                    showactive=True,
                    x=1.25,
                    xanchor="left",
                    y=1,
                    yanchor="top",
                    font={'size': 12}
                ),
            ]
        )
    
    def create_plot(self, output_file: str = 'eeg_ecg_plot.html', decimate_factor: int = 5) -> None:
        """Main method to create the interactive plot."""
        print("Creating EEG/ECG interactive plot...")
        start_time = time.time()
        
        # Load and process data
        self.load_data()
        self.identify_channels()
        
        # Decimate data if needed for performance
        plot_df = self.decimate_data(decimate_factor)
        
        # Create plot structure
        fig = self.create_subplot_structure()
        
        # Add traces
        if self.eeg_channels:
            self.add_eeg_traces(fig, plot_df)
        
        if self.ecg_channels:
            self.add_ecg_traces(fig, plot_df)
        
        if self.other_channels:
            self.add_other_traces(fig, plot_df)
        
        # Configure layout
        self.configure_layout(fig)
        
        # Save to HTML
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'eeg_ecg_plot',
                'height': 900,
                'width': 1200,
                'scale': 2
            }
        }
        
        fig.write_html(output_file, config=config)
        
        print(f"Plot created in {time.time() - start_time:.2f} seconds")
        print(f"Successfully generated interactive plot: {output_file}")
        print(f"Data points plotted: {len(plot_df)}")
        print("Features:")
        print("- Use legend to show/hide channels")
        print("- Use range slider for quick navigation")
        print("- Use toolbar buttons for common views")
        print("- Hover for detailed values")
        print("- Zoom and pan with mouse")


def main():
    """Main execution function."""
    file_path = 'EEG and ECG data_02_raw.csv'
    plotter = EEGPlotter(file_path)
    
    try:
        plotter.create_plot(
            output_file='eeg_ecg_plot.html',
            decimate_factor=3  # Adjust for performance vs quality tradeoff
        )
    except FileNotFoundError:
        print(f"Error: Could not find file '{file_path}'")
        print("Please ensure the CSV file is in the same directory as this script.")
    except Exception as e:
        print(f"Error creating plot: {str(e)}")
        raise


if __name__ == "__main__":
    main()