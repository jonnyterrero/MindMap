const express = require('express');
const { body, validationResult } = require('express-validator');
const dataManager = require('../utils/dataManager');
const router = express.Router();

// Validation middleware
const validateHealthEntry = [
  body('date').isISO8601().withMessage('Date must be in YYYY-MM-DD format'),
  body('sleep.start_time').matches(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/).withMessage('Sleep start time must be in HH:MM format'),
  body('sleep.end_time').matches(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/).withMessage('Sleep end time must be in HH:MM format'),
  body('sleep.quality_score').isInt({ min: 1, max: 10 }).withMessage('Sleep quality score must be between 1 and 10'),
  body('mood.mood_score').isInt({ min: 1, max: 10 }).withMessage('Mood score must be between 1 and 10'),
  body('mood.stress_score').isInt({ min: 1, max: 10 }).withMessage('Stress score must be between 1 and 10'),
  body('symptoms.gi_flare').isInt({ min: 0, max: 10 }).withMessage('GI flare score must be between 0 and 10'),
  body('symptoms.skin_flare').isInt({ min: 0, max: 10 }).withMessage('Skin flare score must be between 0 and 10'),
  body('symptoms.migraine').isInt({ min: 0, max: 10 }).withMessage('Migraine score must be between 0 and 10'),
];

// GET /api/health/entries - Get all health entries
router.get('/entries', async (req, res) => {
  try {
    const { startDate, endDate, limit } = req.query;
    
    let entries;
    if (startDate && endDate) {
      entries = await dataManager.getEntriesForDateRange(startDate, endDate);
    } else {
      entries = await dataManager.loadDataset();
    }

    if (limit) {
      entries = entries.slice(0, parseInt(limit));
    }

    res.json({
      success: true,
      data: entries,
      count: entries.length
    });
  } catch (error) {
    console.error('Error fetching health entries:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch health entries'
    });
  }
});

// GET /api/health/entries/:date - Get entry for specific date
router.get('/entries/:date', async (req, res) => {
  try {
    const { date } = req.params;
    const entry = await dataManager.getEntryForDate(date);
    
    if (!entry) {
      return res.status(404).json({
        success: false,
        error: 'Entry not found for the specified date'
      });
    }

    res.json({
      success: true,
      data: entry
    });
  } catch (error) {
    console.error('Error fetching health entry:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch health entry'
    });
  }
});

// POST /api/health/entries - Create new health entry
router.post('/entries', validateHealthEntry, async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        success: false,
        error: 'Validation failed',
        details: errors.array()
      });
    }

    const {
      date,
      sleep: { start_time, end_time, quality_score },
      mood: { mood_score, stress_score, journal_entry = '', voice_note_path = null },
      symptoms: { gi_flare = 0, skin_flare = 0, migraine = 0 }
    } = req.body;

    const entry = await dataManager.logEntry(
      date,
      start_time,
      end_time,
      quality_score,
      mood_score,
      stress_score,
      journal_entry,
      voice_note_path,
      gi_flare,
      skin_flare,
      migraine
    );

    res.status(201).json({
      success: true,
      data: entry,
      message: 'Health entry created successfully'
    });
  } catch (error) {
    console.error('Error creating health entry:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to create health entry'
    });
  }
});

// PUT /api/health/entries/:date - Update health entry
router.put('/entries/:date', validateHealthEntry, async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        success: false,
        error: 'Validation failed',
        details: errors.array()
      });
    }

    const { date } = req.params;
    const updatedEntry = req.body;

    // Ensure the date in the body matches the URL parameter
    updatedEntry.date = date;

    const entry = await dataManager.updateEntry(date, updatedEntry);

    res.json({
      success: true,
      data: entry,
      message: 'Health entry updated successfully'
    });
  } catch (error) {
    console.error('Error updating health entry:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to update health entry'
    });
  }
});

// DELETE /api/health/entries/:date - Delete health entry
router.delete('/entries/:date', async (req, res) => {
  try {
    const { date } = req.params;
    await dataManager.deleteEntry(date);

    res.json({
      success: true,
      message: 'Health entry deleted successfully'
    });
  } catch (error) {
    console.error('Error deleting health entry:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to delete health entry'
    });
  }
});

// GET /api/health/correlation - Get correlation data for analysis
router.get('/correlation', async (req, res) => {
  try {
    const correlationData = await dataManager.getCorrelationData();
    
    res.json({
      success: true,
      data: correlationData
    });
  } catch (error) {
    console.error('Error fetching correlation data:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch correlation data'
    });
  }
});

// GET /api/health/export - Export all data as JSON
router.get('/export', async (req, res) => {
  try {
    const jsonData = await dataManager.exportData();
    
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Content-Disposition', 'attachment; filename="health_data.json"');
    res.send(jsonData);
  } catch (error) {
    console.error('Error exporting data:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to export data'
    });
  }
});

// POST /api/health/import - Import data from JSON
router.post('/import', async (req, res) => {
  try {
    const { data } = req.body;
    
    if (!data) {
      return res.status(400).json({
        success: false,
        error: 'No data provided'
      });
    }

    const entries = await dataManager.importData(data);
    
    res.json({
      success: true,
      data: entries,
      message: `Successfully imported ${entries.length} entries`
    });
  } catch (error) {
    console.error('Error importing data:', error);
    res.status(500).json({
      success: false,
      error: error.message || 'Failed to import data'
    });
  }
});

// GET /api/health/stats - Get health statistics
router.get('/stats', async (req, res) => {
  try {
    const entries = await dataManager.loadDataset();
    
    if (entries.length === 0) {
      return res.json({
        success: true,
        data: {
          totalEntries: 0,
          averageSleepQuality: 0,
          averageMood: 0,
          averageStress: 0,
          totalSymptoms: 0
        }
      });
    }

    const stats = {
      totalEntries: entries.length,
      averageSleepQuality: entries.reduce((sum, entry) => sum + entry.sleep.quality_score, 0) / entries.length,
      averageMood: entries.reduce((sum, entry) => sum + entry.mood.mood_score, 0) / entries.length,
      averageStress: entries.reduce((sum, entry) => sum + entry.mood.stress_score, 0) / entries.length,
      totalSymptoms: entries.reduce((sum, entry) => 
        sum + entry.symptoms.gi_flare + entry.symptoms.skin_flare + entry.symptoms.migraine, 0
      ),
      dateRange: {
        earliest: entries[0].date,
        latest: entries[entries.length - 1].date
      }
    };

    res.json({
      success: true,
      data: stats
    });
  } catch (error) {
    console.error('Error fetching health stats:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch health statistics'
    });
  }
});

module.exports = router;
