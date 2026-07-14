const fs = require('fs-extra');
const path = require('path');

class DataManager {
  constructor() {
    this.dataDir = process.env.DATA_DIR || './data';
    this.healthLogsFile = path.join(this.dataDir, process.env.HEALTH_LOGS_FILE || 'health_logs.json');
    this.initDataset();
  }

  // Initialize data storage (matches Python init_dataset)
  async initDataset() {
    try {
      await fs.ensureDir(this.dataDir);
      if (!await fs.pathExists(this.healthLogsFile)) {
        await fs.writeFile(this.healthLogsFile, '');
      }
    } catch (error) {
      console.error('Error initializing dataset:', error);
      throw error;
    }
  }

  // Log entry (matches Python log_entry function)
  async logEntry(date, sleepStart, sleepEnd, quality, moodScore, stressScore, 
                 journal = '', voiceNote = null, giFlare = 0, skinFlare = 0, migraine = 0) {
    try {
      // Calculate sleep duration (matches Python logic)
      const durationHours = this.calculateSleepDuration(sleepStart, sleepEnd);

      const entry = {
        date,
        sleep: {
          start_time: sleepStart,
          end_time: sleepEnd,
          duration_hours: Math.round(durationHours * 100) / 100, // Round to 2 decimal places
          quality_score: quality
        },
        mood: {
          mood_score: moodScore,
          stress_score: stressScore,
          journal_entry: journal,
          voice_note_path: voiceNote
        },
        symptoms: {
          gi_flare: giFlare,
          skin_flare: skinFlare,
          migraine: migraine
        }
      };

      // Append to file (matches Python behavior)
      await fs.appendFile(this.healthLogsFile, JSON.stringify(entry) + '\n');
      return entry;
    } catch (error) {
      console.error('Error logging entry:', error);
      throw error;
    }
  }

  // Calculate sleep duration (matches Python logic)
  calculateSleepDuration(startTime, endTime) {
    const [startHour, startMin] = startTime.split(':').map(Number);
    const [endHour, endMin] = endTime.split(':').map(Number);
    
    let startMinutes = startHour * 60 + startMin;
    let endMinutes = endHour * 60 + endMin;
    
    // Handle overnight case (end time is next day)
    if (endMinutes < startMinutes) {
      endMinutes += 24 * 60; // Add 24 hours
    }
    
    const durationMinutes = endMinutes - startMinutes;
    return durationMinutes / 60; // Convert to hours
  }

  // Load dataset (matches Python load_dataset)
  async loadDataset() {
    try {
      const fileContent = await fs.readFile(this.healthLogsFile, 'utf8');
      
      if (!fileContent.trim()) {
        return [];
      }

      // Parse JSONL format (one JSON object per line)
      const lines = fileContent.trim().split('\n');
      const entries = lines.map(line => {
        try {
          return JSON.parse(line);
        } catch (error) {
          console.warn('Invalid JSON line:', line);
          return null;
        }
      }).filter(entry => entry !== null);

      return entries;
    } catch (error) {
      console.error('Error loading dataset:', error);
      return [];
    }
  }

  // Get entries for a specific date
  async getEntryForDate(date) {
    const entries = await this.loadDataset();
    return entries.find(entry => entry.date === date) || null;
  }

  // Get entries for date range
  async getEntriesForDateRange(startDate, endDate) {
    const entries = await this.loadDataset();
    return entries.filter(entry => {
      const entryDate = new Date(entry.date);
      const start = new Date(startDate);
      const end = new Date(endDate);
      return entryDate >= start && entryDate <= end;
    });
  }

  // Update entry for a specific date
  async updateEntry(date, updatedEntry) {
    try {
      const entries = await this.loadDataset();
      const entryIndex = entries.findIndex(entry => entry.date === date);
      
      if (entryIndex !== -1) {
        entries[entryIndex] = updatedEntry;
      } else {
        entries.push(updatedEntry);
      }

      // Sort by date
      entries.sort((a, b) => new Date(a.date) - new Date(b.date));

      // Write back to file
      const content = entries.map(entry => JSON.stringify(entry)).join('\n');
      await fs.writeFile(this.healthLogsFile, content + '\n');
      
      return updatedEntry;
    } catch (error) {
      console.error('Error updating entry:', error);
      throw error;
    }
  }

  // Delete entry for a specific date
  async deleteEntry(date) {
    try {
      const entries = await this.loadDataset();
      const filteredEntries = entries.filter(entry => entry.date !== date);
      
      const content = filteredEntries.map(entry => JSON.stringify(entry)).join('\n');
      await fs.writeFile(this.healthLogsFile, content + (filteredEntries.length > 0 ? '\n' : ''));
      
      return true;
    } catch (error) {
      console.error('Error deleting entry:', error);
      throw error;
    }
  }

  // Get correlation data for analysis
  async getCorrelationData() {
    const entries = await this.loadDataset();
    
    return {
      sleep: entries.map(entry => ({
        duration_hours: entry.sleep.duration_hours,
        quality_score: entry.sleep.quality_score
      })),
      mood: entries.map(entry => ({
        mood_score: entry.mood.mood_score,
        stress_score: entry.mood.stress_score
      })),
      symptoms: entries.map(entry => ({
        gi_flare: entry.symptoms.gi_flare,
        skin_flare: entry.symptoms.skin_flare,
        migraine: entry.symptoms.migraine
      }))
    };
  }

  // Export data as JSON
  async exportData() {
    const entries = await this.loadDataset();
    return JSON.stringify(entries, null, 2);
  }

  // Import data from JSON
  async importData(jsonData) {
    try {
      const entries = JSON.parse(jsonData);
      
      // Validate entries format
      for (const entry of entries) {
        if (!entry.date || !entry.sleep || !entry.mood || !entry.symptoms) {
          throw new Error('Invalid entry format');
        }
      }

      // Write to file
      const content = entries.map(entry => JSON.stringify(entry)).join('\n');
      await fs.writeFile(this.healthLogsFile, content + '\n');
      
      return entries;
    } catch (error) {
      console.error('Error importing data:', error);
      throw new Error('Invalid JSON data format');
    }
  }
}

module.exports = new DataManager();
