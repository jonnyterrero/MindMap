const express = require('express');
const analyticsService = require('../services/analyticsService');
const router = express.Router();

// GET /api/analytics/summary - Get analytics summary
router.get('/summary', async (req, res) => {
  try {
    const result = await analyticsService.getAnalyticsSummary();
    
    if (result.success) {
      res.json(result);
    } else {
      res.status(500).json(result);
    }
  } catch (error) {
    console.error('Analytics summary error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get analytics summary'
    });
  }
});

// POST /api/analytics/correlation - Run correlation analysis
router.post('/correlation', async (req, res) => {
  try {
    const result = await analyticsService.runCorrelationAnalysis();
    
    if (result.success) {
      res.json(result);
    } else {
      res.status(400).json(result);
    }
  } catch (error) {
    console.error('Correlation analysis error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to run correlation analysis'
    });
  }
});

// POST /api/analytics/predict-gi-flare - Run GI flare prediction
router.post('/predict-gi-flare', async (req, res) => {
  try {
    const result = await analyticsService.runGiFlarePrediction();
    
    if (result.success) {
      res.json(result);
    } else {
      res.status(400).json(result);
    }
  } catch (error) {
    console.error('GI flare prediction error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to run GI flare prediction'
    });
  }
});

// POST /api/analytics/comprehensive - Run comprehensive analytics
router.post('/comprehensive', async (req, res) => {
  try {
    const result = await analyticsService.runComprehensiveAnalytics();
    
    if (result.success) {
      res.json(result);
    } else {
      res.status(400).json(result);
    }
  } catch (error) {
    console.error('Comprehensive analytics error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to run comprehensive analytics'
    });
  }
});

// GET /api/analytics/insights - Get generated insights
router.get('/insights', async (req, res) => {
  try {
    const result = await analyticsService.runComprehensiveAnalytics();
    
    if (result.success && result.data.overall_insights) {
      res.json({
        success: true,
        data: {
          insights: result.data.overall_insights,
          generated_at: result.data.generated_at
        }
      });
    } else {
      res.status(400).json({
        success: false,
        error: 'No insights available. Ensure you have sufficient data (minimum 3 entries for correlation, 10 for predictions).'
      });
    }
  } catch (error) {
    console.error('Insights generation error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to generate insights'
    });
  }
});

module.exports = router;
