const dataManager = require('../utils/dataManager');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs-extra');

class AnalyticsService {
  constructor() {
    this.pythonScriptPath = path.join(__dirname, '../scripts/ml_analytics.py');
  }

  // Run correlation analysis
  async runCorrelationAnalysis() {
    try {
      const entries = await dataManager.loadDataset();
      
      if (entries.length < 3) {
        return {
          success: false,
          error: 'Insufficient data for correlation analysis (minimum 3 entries required)',
          data: null
        };
      }

      // Convert to DataFrame-like structure for Python
      const dfData = entries.map(entry => ({
        'sleep.duration_hours': entry.sleep.duration_hours,
        'sleep.quality_score': entry.sleep.quality_score,
        'mood.mood_score': entry.mood.mood_score,
        'mood.stress_score': entry.mood.stress_score,
        'symptoms.gi_flare': entry.symptoms.gi_flare,
        'symptoms.skin_flare': entry.symptoms.skin_flare,
        'symptoms.migraine': entry.symptoms.migraine
      }));

      // Run Python correlation analysis
      const correlationResult = await this.runPythonScript('correlation', dfData);
      
      return {
        success: true,
        data: {
          correlation_matrix: correlationResult.correlation_matrix,
          insights: this.generateCorrelationInsights(correlationResult.correlation_matrix),
          data_points: entries.length
        }
      };
    } catch (error) {
      console.error('Correlation analysis error:', error);
      return {
        success: false,
        error: error.message,
        data: null
      };
    }
  }

  // Run GI flare prediction model
  async runGiFlarePrediction() {
    try {
      const entries = await dataManager.loadDataset();
      
      if (entries.length < 10) {
        return {
          success: false,
          error: 'Insufficient data for prediction model (minimum 10 entries required)',
          data: null
        };
      }

      // Convert to DataFrame-like structure for Python
      const dfData = entries.map(entry => ({
        'sleep.duration_hours': entry.sleep.duration_hours,
        'sleep.quality_score': entry.sleep.quality_score,
        'mood.mood_score': entry.mood.mood_score,
        'mood.stress_score': entry.mood.stress_score,
        'symptoms.gi_flare': entry.symptoms.gi_flare
      }));

      // Run Python prediction model
      const predictionResult = await this.runPythonScript('predict_gi_flare', dfData);
      
      return {
        success: true,
        data: {
          model_coefficients: predictionResult.coefficients,
          intercept: predictionResult.intercept,
          mse: predictionResult.mse,
          r2_score: predictionResult.r2_score,
          feature_importance: this.calculateFeatureImportance(predictionResult.coefficients),
          predictions: this.generatePredictionInsights(predictionResult),
          data_points: entries.length
        }
      };
    } catch (error) {
      console.error('GI flare prediction error:', error);
      return {
        success: false,
        error: error.message,
        data: null
      };
    }
  }

  // Run comprehensive analytics
  async runComprehensiveAnalytics() {
    try {
      const [correlationResult, predictionResult] = await Promise.all([
        this.runCorrelationAnalysis(),
        this.runGiFlarePrediction()
      ]);

      return {
        success: true,
        data: {
          correlation_analysis: correlationResult.success ? correlationResult.data : null,
          gi_flare_prediction: predictionResult.success ? predictionResult.data : null,
          overall_insights: this.generateOverallInsights(correlationResult, predictionResult),
          generated_at: new Date().toISOString()
        }
      };
    } catch (error) {
      console.error('Comprehensive analytics error:', error);
      return {
        success: false,
        error: error.message,
        data: null
      };
    }
  }

  // Run Python script
  async runPythonScript(analysisType, data) {
    return new Promise((resolve, reject) => {
      const pythonProcess = spawn('python3', [this.pythonScriptPath, '--type', analysisType], {
        stdio: ['pipe', 'pipe', 'pipe']
      });

      let output = '';
      let errorOutput = '';

      pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(output);
            resolve(result);
          } catch (error) {
            reject(new Error('Failed to parse Python script output'));
          }
        } else {
          reject(new Error(`Python script failed: ${errorOutput}`));
        }
      });

      // Send data to Python script
      pythonProcess.stdin.write(JSON.stringify(data));
      pythonProcess.stdin.end();
    });
  }

  // Generate correlation insights
  generateCorrelationInsights(correlationMatrix) {
    const insights = [];
    
    // Sleep-Mood correlation
    const sleepMoodCorr = correlationMatrix['sleep.quality_score']['mood.mood_score'];
    if (Math.abs(sleepMoodCorr) > 0.5) {
      insights.push({
        type: 'correlation',
        title: 'Sleep-Mood Connection',
        description: `Strong ${sleepMoodCorr > 0 ? 'positive' : 'negative'} correlation (${sleepMoodCorr.toFixed(2)}) between sleep quality and mood`,
        confidence: Math.abs(sleepMoodCorr),
        actionable: true
      });
    }

    // Stress-Symptoms correlation
    const stressGiCorr = correlationMatrix['mood.stress_score']['symptoms.gi_flare'];
    if (Math.abs(stressGiCorr) > 0.5) {
      insights.push({
        type: 'correlation',
        title: 'Stress-GI Flare Connection',
        description: `Strong ${stressGiCorr > 0 ? 'positive' : 'negative'} correlation (${stressGiCorr.toFixed(2)}) between stress and GI flares`,
        confidence: Math.abs(stressGiCorr),
        actionable: true
      });
    }

    // Sleep-Symptoms correlation
    const sleepGiCorr = correlationMatrix['sleep.quality_score']['symptoms.gi_flare'];
    if (Math.abs(sleepGiCorr) > 0.5) {
      insights.push({
        type: 'correlation',
        title: 'Sleep-GI Flare Connection',
        description: `Strong ${sleepGiCorr > 0 ? 'positive' : 'negative'} correlation (${sleepGiCorr.toFixed(2)}) between sleep quality and GI flares`,
        confidence: Math.abs(sleepGiCorr),
        actionable: true
      });
    }

    return insights;
  }

  // Calculate feature importance
  calculateFeatureImportance(coefficients) {
    const features = [
      'sleep.duration_hours',
      'sleep.quality_score',
      'mood.mood_score',
      'mood.stress_score'
    ];

    return features.map((feature, index) => ({
      feature,
      coefficient: coefficients[index],
      importance: Math.abs(coefficients[index])
    })).sort((a, b) => b.importance - a.importance);
  }

  // Generate prediction insights
  generatePredictionInsights(predictionResult) {
    const insights = [];
    
    // Model performance
    if (predictionResult.mse < 2.0) {
      insights.push({
        type: 'model_performance',
        title: 'Good Model Performance',
        description: `The GI flare prediction model has good accuracy (MSE: ${predictionResult.mse.toFixed(2)})`,
        confidence: 0.8,
        actionable: false
      });
    }

    // Feature importance
    const featureImportance = this.calculateFeatureImportance(predictionResult.coefficients);
    const topFeature = featureImportance[0];
    
    insights.push({
      type: 'feature_importance',
      title: 'Most Important Factor',
      description: `${topFeature.feature.replace('.', ' ').replace('_', ' ')} is the most important factor for GI flare prediction`,
      confidence: 0.7,
      actionable: true
    });

    return insights;
  }

  // Generate overall insights
  generateOverallInsights(correlationResult, predictionResult) {
    const insights = [];

    if (correlationResult.success && correlationResult.data) {
      insights.push(...correlationResult.data.insights);
    }

    if (predictionResult.success && predictionResult.data) {
      insights.push(...predictionResult.data.predictions);
    }

    // Sort by confidence
    insights.sort((a, b) => b.confidence - a.confidence);

    return insights;
  }

  // Get analytics summary
  async getAnalyticsSummary() {
    try {
      const entries = await dataManager.loadDataset();
      
      if (entries.length === 0) {
        return {
          success: true,
          data: {
            total_entries: 0,
            analysis_available: false,
            message: 'No data available for analysis'
          }
        };
      }

      const summary = {
        total_entries: entries.length,
        date_range: {
          earliest: entries[0].date,
          latest: entries[entries.length - 1].date
        },
        analysis_available: entries.length >= 3,
        prediction_available: entries.length >= 10,
        average_metrics: {
          sleep_quality: entries.reduce((sum, entry) => sum + entry.sleep.quality_score, 0) / entries.length,
          mood_score: entries.reduce((sum, entry) => sum + entry.mood.mood_score, 0) / entries.length,
          stress_score: entries.reduce((sum, entry) => sum + entry.mood.stress_score, 0) / entries.length,
          gi_flare: entries.reduce((sum, entry) => sum + entry.symptoms.gi_flare, 0) / entries.length
        }
      };

      return {
        success: true,
        data: summary
      };
    } catch (error) {
      console.error('Analytics summary error:', error);
      return {
        success: false,
        error: error.message,
        data: null
      };
    }
  }
}

module.exports = new AnalyticsService();
