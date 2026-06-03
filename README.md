 # VIBE                                                                                                                                      
                                                                                                                                                                  
  Predict a continuous behavioral stress score (z-score) from four implanted-sensor                                                                               
  physiological signals in rats, and apply the trained model to a continuous test                                                                                 
  recording with a sliding window.                                                                                                                                
                                                                                                                                                                  
  ## Overview                                                                                                                                                     
                                                                                                                                                                  
  Each training recording yields one labeled example: multi-channel                                                                                   
  physiology paired with a single behavioral z-score. A regression model is
  trained on hand-engineered statistical features and evaluated with leave-one-out cross-validation across recordings. The selected model is then re-fit on all                                                                                    
  training data and applied to a held-out continuous test recording in sliding                                                                                    
  windows to produce a time-resolved stress trace.                                                                                                                
                                                                                                                                                                  
  Signals used (per recording):                                                                                                                                   
  - Blood Pressure (mmHg)                                                                                                                                         
  - Heart Rate (BPM)
  - Glucose (mg/dL)                                                                                                                                               
  - Breathing Period (s)

  ## Repository layout                                                                                                                                            
                  
  ```
  .
  ├── main.py                      # End-to-end pipeline entry point
  ├── config.py                    # Paths, model list, sliding-window settings                                                                                   
  ├── dataInterface/
  │   ├── dataLoader.py            # Reads training xlsx + continuous test xlsx                                                                                   
  │   └── featureExtractor.py      # Per-signal summary statistics (10 per channel)                                                                               
  ├── machineLearning/                                                                                                                                            
  │   ├── models.py                # Model registry                                                                                                       
  │   ├── preprocessing.py         # Leave-one-out splitter                                                                                                       
  │   ├── trainer.py               # CV evaluation + final fit
  │   └── predictor.py             # Inference                                                                                        
  ├── plottingHelpers/                                                                                                                                            
  │   └── plotter.py               # Figures plotting                                                                                       
  ├── data/                                                                                                                                                       
  │   ├── raw/                     # Per-recording xlsx training files
  │   └── test_data.xlsx           # Continuous multi-channel test recording                                                                                      
  ├── figures/                     # Output PDFs                                                                                                
  ├── artifacts                    # Pickled model + scaler + feature columns                                                                                     
  └── test_predictions.xlsx        # Sliding-window predictions on test data                                                                                      
  ```                                                                                                                                                             
  
  ## System Requirements

  ### Hardware Requirements

  The `VIBE` package requires only a standard computer with enough RAM to support the operations.

  ### Software Requirements

  #### OS Requirements
  This package is supported for *macOS* and *Windows*. The package has been tested on the following systems:
  + macOS: Tahoe (26.5)
  + Windows: 
  
  All dependencies are listed in `requirements.txt`. Tested on Windows and macOS.
   
  ## 🔧 Installation Guide
  
  Installation times on standard computers will require approximately 1 minute.

  ```bash
  # 1. Clone the repository
  git clone https://github.com/RLiuJ/VIBE.git

  # 2. Move into the project directory
  cd VIBE
  
  # 3. Install dependencies
  pip install -r Requirements.txt
  ```
  
  ## Demo
  
  Prediction results for test data will require under 1 minute run time.
  
  ### Running                                                                                                                                                      
                  
  ```bash
  python main.py
  ```                                                                                 

  ### Outputs
  - `artifacts`.
  - `test_predictions.xlsx`.                                                     
  - `figures/`.                                                                                                                                       
                  
  ## Instructions for Use

  ### `main.py`
   
  The pipeline (1) loads all `data/raw/*.xlsx`, (2) builds the feature matrix,                                                                                    
  (3) runs LOO-CV for every model in `Config.modelNames`, (4) re-fits
  `Config.bestModel` on all data and saves `artifacts`, (5) runs sliding-window 
  inference on `data/test_data.xlsx`, (6) writes `test_predictions.xlsx`, and
  (7) renders all figures.                                                                                                                  

  ### Evaluation

  - Leave-one-out CV across recordings (`machineLearning/preprocessing.py`). 
  - Reported metrics: `MAE`, Pearson `r`, `R²`, sign accuracy, three-class accuracy.                                                                              
  - Three-class buckets: `healthy` (z ≥ 0), `lowStress` (-2 < z < 0),                                                                                             
    `highStress` (z ≤ -2). Thresholds defined in `trainer.py` and `predictor.py`.      

  ## License

  This repository is licensed under the MIT License. For research use only. Please cite our work if used in academic publications.
