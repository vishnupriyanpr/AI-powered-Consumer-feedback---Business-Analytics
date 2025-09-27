// AMIL Project JavaScript - AI Customer Feedback Analyzer
// Optimized for RTX 4060 GPU acceleration

class FeedbackAnalyzer {
    constructor() {
        this.apiBaseUrl = 'http://localhost:5000/api';
        this.currentTab = 'analyzer';
        this.charts = {};
        this.analysisHistory = [];

        this.initializeApp();
    }

    initializeApp() {
        this.setupEventListeners();
        this.initializeCharts();
        this.checkGPUStatus();
        this.loadAnalyticsData();
    }

    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchTab(e.target.closest('.nav-btn').dataset.tab);
            });
        });

        // Analyze button
        document.getElementById('analyzeBtn').addEventListener('click', () => {
            this.analyzeFeedback();
        });

        // Real-time input validation
        document.getElementById('feedbackInput').addEventListener('input', (e) => {
            this.validateInput(e.target.value);
        });

        // Template selection
        document.querySelectorAll('.template-item').forEach(item => {
            item.addEventListener('click', () => {
                this.applyTemplate(item);
            });
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                this.analyzeFeedback();
            }
        });
    }

    switchTab(tabName) {
        // Update navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(tabName).classList.add('active');

        this.currentTab = tabName;

        // Refresh charts if analytics tab
        if (tabName === 'analytics') {
            setTimeout(() => this.refreshCharts(), 100);
        }
    }

    async analyzeFeedback() {
        const feedbackText = document.getElementById('feedbackInput').value.trim();

        if (!feedbackText) {
            this.showAlert('Please enter feedback text to analyze', 'warning');
            return;
        }

        this.showLoading(true);
        const startTime = Date.now();

        try {
            const analysisData = {
                text: feedbackText,
                language: document.getElementById('languageSelect').value,
                timestamp: new Date().toISOString()
            };

            // Parallel API calls for better performance
            const [sentimentResult, themesResult, urgencyResult, responseResult] = await Promise.all([
                this.callAPI('/analyze/sentiment', analysisData),
                this.callAPI('/analyze/themes', analysisData),
                this.callAPI('/analyze/urgency', analysisData),
                this.callAPI('/generate/response', analysisData)
            ]);

            const inferenceTime = Date.now() - startTime;

            // Update UI with results
            this.displayResults({
                sentiment: sentimentResult,
                themes: themesResult,
                urgency: urgencyResult,
                response: responseResult,
                inferenceTime: inferenceTime
            });

            // Generate GDG actions
            await this.generateGDGActions({
                sentiment: sentimentResult,
                themes: themesResult,
                urgency: urgencyResult
            });

            // Store in history
            this.analysisHistory.push({
                ...analysisData,
                results: {
                    sentiment: sentimentResult,
                    themes: themesResult,
                    urgency: urgencyResult,
                    response: responseResult
                },
                inferenceTime: inferenceTime
            });

            // Update analytics
            this.updateAnalytics();

        } catch (error) {
            console.error('Analysis failed:', error);
            this.showAlert('Analysis failed. Please check your connection and try again.', 'error');
        } finally {
            this.showLoading(false);
        }
    }

    async callAPI(endpoint, data) {
        const response = await fetch(`${this.apiBaseUrl}${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error(`API call failed: ${response.status}`);
        }

        return await response.json();
    }

    displayResults(results) {
        // Sentiment Display
        const sentimentDisplay = document.getElementById('sentimentDisplay');
        const sentimentScore = results.sentiment.score || 0;
        const sentimentLabel = results.sentiment.label || 'Neutral';

        sentimentDisplay.querySelector('.sentiment-score').textContent =
            (sentimentScore * 100).toFixed(1);
        sentimentDisplay.querySelector('.sentiment-label').textContent = sentimentLabel;

        // Apply color based on sentiment
        const scoreElement = sentimentDisplay.querySelector('.sentiment-score');
        scoreElement.className = `sentiment-score ${sentimentLabel.toLowerCase()}`;

        // Urgency Display
        const urgencyDisplay = document.getElementById('urgencyDisplay');
        const urgencyScore = results.urgency.score || 0;
        const urgencyLevel = results.urgency.level || 'Low';

        urgencyDisplay.querySelector('.urgency-fill').style.width = `${urgencyScore * 100}%`;
        urgencyDisplay.querySelector('.urgency-text').textContent =
            `${urgencyLevel} Priority (${(urgencyScore * 100).toFixed(1)}%)`;

        // Themes Display
        const themesDisplay = document.getElementById('themesDisplay');
        if (results.themes.topics && results.themes.topics.length > 0) {
            themesDisplay.innerHTML = results.themes.topics.map(theme =>
                `<span class="theme-tag">${theme}</span>`
            ).join('');
        } else {
            themesDisplay.innerHTML = '<div class="no-data">No themes detected</div>';
        }

        // Response Display
        const responseDisplay = document.getElementById('responseDisplay');
        if (results.response.text) {
            responseDisplay.textContent = results.response.text;
        } else {
            responseDisplay.innerHTML = '<div class="no-data">No response generated</div>';
        }

        // Performance Metrics
        document.getElementById('inferenceTime').textContent = `${results.inferenceTime}ms`;
        document.getElementById('modelConfidence').textContent =
            `${(results.sentiment.confidence * 100).toFixed(1)}%`;
    }

    async generateGDGActions(analysisResults) {
        try {
            const actionsResult = await this.callAPI('/generate/actions', {
                sentiment: analysisResults.sentiment,
                themes: analysisResults.themes,
                urgency: analysisResults.urgency
            });

            const actionList = document.getElementById('actionList');

            if (actionsResult.actions && actionsResult.actions.length > 0) {
                actionList.innerHTML = actionsResult.actions.map(action => `
                    <div class="action-item">
                        <div class="action-title">${action.title}</div>
                        <div class="action-description">${action.description}</div>
                        <div class="action-priority ${action.priority.toLowerCase()}">${action.priority} Priority</div>
                    </div>
                `).join('');
            } else {
                actionList.innerHTML = '<div class="no-actions">No specific actions recommended</div>';
            }
        } catch (error) {
            console.error('Failed to generate GDG actions:', error);
        }
    }

    initializeCharts() {
        // Sentiment Chart
        const sentimentCtx = document.getElementById('sentimentChart').getContext('2d');
        this.charts.sentiment = new Chart(sentimentCtx, {
            type: 'doughnut',
            data: {
                labels: ['Positive', 'Neutral', 'Negative'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: ['#10b981', '#64748b', '#ef4444'],
                    borderColor: ['#059669', '#475569', '#dc2626'],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        labels: { color: '#cbd5e1' }
                    }
                }
            }
        });

        // Theme Chart
        const themeCtx = document.getElementById('themeChart').getContext('2d');
        this.charts.themes = new Chart(themeCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Frequency',
                    data: [],
                    backgroundColor: '#2563eb',
                    borderColor: '#1d4ed8',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { color: '#cbd5e1' },
                        grid: { color: '#334155' }
                    },
                    x: {
                        ticks: { color: '#cbd5e1' },
                        grid: { color: '#334155' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#cbd5e1' }
                    }
                }
            }
        });

        // Urgency Timeline Chart
        const urgencyCtx = document.getElementById('urgencyChart').getContext('2d');
        this.charts.urgency = new Chart(urgencyCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Urgency Score',
                    data: [],
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        ticks: { color: '#cbd5e1' },
                        grid: { color: '#334155' }
                    },
                    x: {
                        ticks: { color: '#cbd5e1' },
                        grid: { color: '#334155' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#cbd5e1' }
                    }
                }
            }
        });
    }

    updateAnalytics() {
        if (this.analysisHistory.length === 0) return;

        // Update sentiment distribution
        const sentimentCounts = { positive: 0, neutral: 0, negative: 0 };
        const themeCounts = {};
        const urgencyData = [];

        this.analysisHistory.forEach((item, index) => {
            const sentiment = item.results.sentiment.label.toLowerCase();
            sentimentCounts[sentiment]++;

            // Count themes
            if (item.results.themes.topics) {
                item.results.themes.topics.forEach(theme => {
                    themeCounts[theme] = (themeCounts[theme] || 0) + 1;
                });
            }

            // Urgency timeline
            urgencyData.push({
                x: index + 1,
                y: item.results.urgency.score
            });
        });

        // Update sentiment chart
        this.charts.sentiment.data.datasets[0].data = [
            sentimentCounts.positive,
            sentimentCounts.neutral,
            sentimentCounts.negative
        ];
        this.charts.sentiment.update();

        // Update theme chart
        const topThemes = Object.entries(themeCounts)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 5);

        this.charts.themes.data.labels = topThemes.map(([theme]) => theme);
        this.charts.themes.data.datasets[0].data = topThemes.map(([,count]) => count);
        this.charts.themes.update();

        // Update urgency chart
        this.charts.urgency.data.labels = urgencyData.map((_, i) => `Analysis ${i + 1}`);
        this.charts.urgency.data.datasets[0].data = urgencyData.map(item => item.y);
        this.charts.urgency.update();

        // Update statistics
        this.updateStatistics();
    }

    updateStatistics() {
        const totalFeedback = this.analysisHistory.length;
        const positiveFeedback = this.analysisHistory.filter(
            item => item.results.sentiment.label.toLowerCase() === 'positive'
        ).length;
        const highPriority = this.analysisHistory.filter(
            item => item.results.urgency.score > 0.8
        ).length;

        const positiveRate = totalFeedback > 0 ? (positiveFeedback / totalFeedback * 100) : 0;

        const statsGrid = document.getElementById('statsGrid');
        const statItems = statsGrid.querySelectorAll('.stat-item');

        statItems[0].querySelector('.stat-value').textContent = totalFeedback;
        statItems[1].querySelector('.stat-value').textContent = `${positiveRate.toFixed(1)}%`;
        statItems[2].querySelector('.stat-value').textContent = highPriority;
    }

    refreshCharts() {
        Object.values(this.charts).forEach(chart => {
            chart.resize();
            chart.update();
        });
    }

    async checkGPUStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/system/gpu-status`);
            const data = await response.json();

            const gpuStatus = document.getElementById('gpuStatus');
            const gpuAcceleration = document.getElementById('gpuAcceleration');

            if (data.gpu_available) {
                gpuStatus.innerHTML = `<i class="fas fa-microchip"></i><span>${data.gpu_name}</span>`;
                gpuAcceleration.textContent = 'Enabled';
            } else {
                gpuStatus.innerHTML = `<i class="fas fa-exclamation-triangle"></i><span>CPU Fallback</span>`;
                gpuAcceleration.textContent = 'Disabled';
            }
        } catch (error) {
            console.warn('Could not check GPU status:', error);
        }
    }

    async loadAnalyticsData() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/analytics/history`);
            const data = await response.json();

            if (data.history && data.history.length > 0) {
                this.analysisHistory = data.history;
                this.updateAnalytics();
            }
        } catch (error) {
            console.warn('Could not load analytics data:', error);
        }
    }

    validateInput(text) {
        const analyzeBtn = document.getElementById('analyzeBtn');
        const isValid = text.trim().length > 10;

        analyzeBtn.disabled = !isValid;
        analyzeBtn.style.opacity = isValid ? '1' : '0.6';
    }

    showLoading(show) {
        const loadingOverlay = document.getElementById('loadingOverlay');
        loadingOverlay.classList.toggle('active', show);
    }

    showAlert(message, type = 'info') {
        // Create alert element
        const alert = document.createElement('div');
        alert.className = `alert alert-${type}`;
        alert.innerHTML = `
            <div class="alert-content">
                <i class="fas fa-${type === 'error' ? 'exclamation-circle' :
                                   type === 'warning' ? 'exclamation-triangle' :
                                   'info-circle'}"></i>
                <span>${message}</span>
            </div>
        `;

        // Add to DOM
        document.body.appendChild(alert);

        // Auto remove after 5 seconds
        setTimeout(() => {
            alert.remove();
        }, 5000);
    }

    applyTemplate(templateElement) {
        const title = templateElement.querySelector('.template-title').textContent;
        const description = templateElement.querySelector('.template-description').textContent;

        // Add to action list
        const actionList = document.getElementById('actionList');
        const actionItem = document.createElement('div');
        actionItem.className = 'action-item';
        actionItem.innerHTML = `
            <div class="action-title">${title}</div>
            <div class="action-description">${description}</div>
            <div class="action-priority medium">Medium Priority</div>
        `;

        if (actionList.querySelector('.no-actions')) {
            actionList.innerHTML = '';
        }

        actionList.appendChild(actionItem);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new FeedbackAnalyzer();
});

// Service Worker for offline functionality
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js')
        .then(registration => console.log('SW registered'))
        .catch(error => console.log('SW registration failed'));
}
