 # VIBE                                                                                                                                      
                                                                                                                                                                  
  Predict a continuous behavioral stress score (z-score) from four implanted-sensor                                                                               
  physiological signals in rats, and apply the trained model to a continuous test                                                                                 
  recording with a sliding window.                                                                                                                                
                                                                                                                                                                  
  ## Overview                                                                                                                                                     
                                                                                                                                                                  
  Each training recording yields one labeled example: multi-channel                                                                                   
  physiology paired with a single behavioral z-score. A regression model is
  trained on hand-engineered statistical features and evaluated with leave-one-out                                                                                
  cross-validation across recordings. The selected model is then re-fit on all                                                                                    
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
   
  ## Evaluation

  - Leave-one-out CV across recordings (`machineLearning/preprocessing.py`). 
  - Reported metrics: `MAE`, Pearson `r`, `R²`, sign accuracy, three-class accuracy.                                                                              
  - Three-class buckets: `healthy` (z ≥ 0), `lowStress` (-2 < z < 0),                                                                                             
    `highStress` (z ≤ -2). Thresholds defined in `trainer.py` and `predictor.py`.                                                                                 

  ## Outputs
  - `artifacts` — joblib-pickled `{scaler, model, featureColumns}`.
  - `test_predictions.xlsx` — per-window `tStart, tCenter, tEnd, zPred,                                                                                           
    zClipped, zSmoothed, class`.                                                                                                                                  
  - `figures/`:                                                                                                                                                   
    - `fig1_training_performance.pdf`                                                                                                                             
    - `fig2_test_prediction.pdf`                                                                                                                                  
    - `fig3_feature_analysis.pdf`
    - `fig4_model_comparison.pdf`                                                                                                                                 
    - `fig5_per_sample_analysis.pdf`                                                                                                                              
    - `shap_beeswarm_comparison.pdf`
    - `shap_decision.pdf`                                                                                                                                         
                  
  ## Running                                                                                                                                                      
                  
  ```bash
  python main.py
  ```                                                                                                                                                             
   
  The pipeline (1) loads all `data/raw/*.xlsx`, (2) builds the feature matrix,                                                                                    
  (3) runs LOO-CV for every model in `Config.modelNames`, (4) re-fits
  `Config.bestModel` on all data and saves `artifacts`, (5) runs sliding-window                                                                                   
  inference on `data/test_data.xlsx`, (6) writes `test_predictions.xlsx`, and
  (7) renders all figures.                                                                                                                  
                                                                                                                                                                  

