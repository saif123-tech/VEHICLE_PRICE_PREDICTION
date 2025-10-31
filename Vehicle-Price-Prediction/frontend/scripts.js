// Configuration
const API_BASE_URL = 'http://127.0.0.1:8000';
const LOADING_DURATION = 3000; // 3 seconds

// DOM Elements
const loadingScreen = document.getElementById('loading-screen');
const mainApp = document.getElementById('main-app');
const predictionForm = document.getElementById('prediction-form');
const predictBtn = document.getElementById('predict-btn');
const resultCard = document.getElementById('result-card');
const resultContent = document.getElementById('result-content');
const chartContent = document.getElementById('chart-content');
const advancedToggle = document.getElementById('advanced-toggle');
const advancedContent = document.getElementById('advanced-content');
const errorModal = document.getElementById('error-modal');
const errorMessage = document.getElementById('error-message');
const errorModalClose = document.getElementById('error-modal-close');
const errorModalOk = document.getElementById('error-modal-ok');

// Feature importance data (will be fetched from API or use default)
let featureImportanceData = [
    { feature: 'transmission_Manual', importance: 0.1758 },
    { feature: 'max_power_bhp', importance: 0.1692 },
    { feature: 'make_None', importance: 0.0900 },
    { feature: 'transmission_Automatic', importance: 0.0578 },
    { feature: 'owner_0', importance: 0.0507 },
    { feature: 'make_Isuzu', importance: 0.0466 },
    { feature: 'fuel_Petrol', importance: 0.0372 },
    { feature: 'engine_cc', importance: 0.0351 },
    { feature: 'fuel_None', importance: 0.0334 },
    { feature: 'age', importance: 0.0299 }
];

// Initialize the application
document.addEventListener('DOMContentLoaded', function () {
    initializeApp();
});

async function initializeApp() {
    // Show loading screen
    showLoadingScreen();

    // Check API connection
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            const healthData = await response.json();
            console.log('API Health:', healthData);
            updateModelInfo(healthData);
        }
    } catch (error) {
        console.warn('API not available, using offline mode');
    }

    // Hide loading screen after duration
    setTimeout(() => {
        hideLoadingScreen();
        initializeEventListeners();
        initializeAnimations();
    }, LOADING_DURATION);
}

function showLoadingScreen() {
    loadingScreen.style.display = 'flex';
    mainApp.classList.add('hidden');
}

function hideLoadingScreen() {
    loadingScreen.style.opacity = '0';
    setTimeout(() => {
        loadingScreen.style.display = 'none';
        mainApp.classList.remove('hidden');
    }, 500);
}

function updateModelInfo(healthData) {
    const modelNameElement = document.getElementById('model-name');
    if (modelNameElement && healthData.model_status === 'loaded') {
        modelNameElement.textContent = 'XGBRegressor';
    }
}

function initializeEventListeners() {
    // Form submission
    predictionForm.addEventListener('submit', handleFormSubmit);

    // Advanced options toggle
    advancedToggle.addEventListener('click', toggleAdvancedOptions);

    // Modal close events
    errorModalClose.addEventListener('click', hideErrorModal);
    errorModalOk.addEventListener('click', hideErrorModal);

    // Click outside modal to close
    errorModal.addEventListener('click', function (e) {
        if (e.target === errorModal) {
            hideErrorModal();
        }
    });

    // Input animations
    const inputs = document.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.addEventListener('focus', function () {
            this.parentElement.classList.add('focused');
        });

        input.addEventListener('blur', function () {
            this.parentElement.classList.remove('focused');
        });
    });
}

function initializeAnimations() {
    // Animate cards on scroll
    const cards = document.querySelectorAll('.card');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.animationDelay = `${Math.random() * 0.3}s`;
                entry.target.classList.add('animate-in');
            }
        });
    }, { threshold: 0.1 });

    cards.forEach(card => observer.observe(card));
}

function toggleAdvancedOptions() {
    const isActive = advancedContent.classList.contains('active');

    if (isActive) {
        advancedContent.classList.remove('active');
        advancedToggle.classList.remove('active');
    } else {
        advancedContent.classList.add('active');
        advancedToggle.classList.add('active');
    }
}

async function handleFormSubmit(e) {
    e.preventDefault();

    // Show loading state
    predictBtn.classList.add('loading');
    predictBtn.disabled = true;

    try {
        // Collect form data
        const formData = new FormData(predictionForm);
        const carFeatures = {};

        // Convert form data to object
        for (let [key, value] of formData.entries()) {
            // Convert numeric fields
            if (['year', 'engine_cc', 'km_driven', 'seats'].includes(key)) {
                carFeatures[key] = parseInt(value);
            } else if (['max_power_bhp', 'mileage_value', 'torque_nm', 'torque_rpm'].includes(key)) {
                carFeatures[key] = parseFloat(value);
            } else {
                carFeatures[key] = value;
            }
        }

        // Add default values for required fields
        carFeatures.mileage_unit = 'kmpl';

        console.log('Sending prediction request:', carFeatures);

        // Make API request
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(carFeatures)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log('Prediction result:', result);

        // Display results
        displayPredictionResult(result);
        displayFeatureImportance();

        // Scroll to results
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (error) {
        console.error('Prediction error:', error);
        showErrorModal('Failed to get prediction. Please check if the API server is running.');
    } finally {
        // Hide loading state
        predictBtn.classList.remove('loading');
        predictBtn.disabled = false;
    }
}

function displayPredictionResult(result) {
    const price = result.predicted_price;
    const formattedPrice = result.formatted_price || `₹${price.toLocaleString()}`;
    const category = result.price_category || getPriceCategory(price);
    const confidence = result.confidence_level || getConfidenceLevel(price);

    // Create result HTML
    const resultHTML = `
        <div class="price-display">
            <div class="price-amount">${formattedPrice}</div>
            <div class="price-category" style="color: ${getCategoryColor(category)}">${category}</div>
            <div class="price-details">
                <div class="price-detail">
                    <div class="price-detail-value">${confidence}</div>
                    <div class="price-detail-label">Confidence Level</div>
                </div>
                <div class="price-detail">
                    <div class="price-detail-value">${result.model_used || 'XGBRegressor'}</div>
                    <div class="price-detail-label">AI Model</div>
                </div>
            </div>
            <div class="prediction-info">
                <p style="margin-top: 1rem; color: #666; font-size: 0.9rem;">
                    <i class="fas fa-info-circle"></i>
                    Prediction made on ${new Date().toLocaleDateString()} using advanced machine learning algorithms.
                </p>
            </div>
        </div>
    `;

    resultContent.innerHTML = resultHTML;

    // Animate the price display
    setTimeout(() => {
        const priceAmount = resultContent.querySelector('.price-amount');
        if (priceAmount) {
            animateNumber(priceAmount, 0, price, 1000);
        }
    }, 300);
}

function displayFeatureImportance() {
    const chartHTML = `
        <div class="feature-chart">
            ${featureImportanceData.slice(0, 8).map(item => `
                <div class="feature-item">
                    <div class="feature-name">${formatFeatureName(item.feature)}</div>
                    <div class="feature-bar">
                        <div class="feature-bar-fill" style="--width: ${item.importance * 100}%; width: 0;"></div>
                    </div>
                    <div class="feature-value">${(item.importance * 100).toFixed(1)}%</div>
                </div>
            `).join('')}
        </div>
    `;

    chartContent.innerHTML = chartHTML;

    // Animate bars
    setTimeout(() => {
        const bars = chartContent.querySelectorAll('.feature-bar-fill');
        bars.forEach((bar, index) => {
            setTimeout(() => {
                const width = getComputedStyle(bar).getPropertyValue('--width');
                bar.style.width = width;
            }, index * 100);
        });
    }, 500);
}

function formatFeatureName(feature) {
    const featureMap = {
        'transmission_Manual': 'Manual Transmission',
        'max_power_bhp': 'Engine Power (BHP)',
        'make_None': 'Brand Recognition',
        'transmission_Automatic': 'Automatic Transmission',
        'owner_0': 'First Owner',
        'make_Isuzu': 'Isuzu Brand',
        'fuel_Petrol': 'Petrol Fuel',
        'engine_cc': 'Engine Displacement',
        'fuel_None': 'Fuel Type',
        'age': 'Vehicle Age'
    };

    return featureMap[feature] || feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function getPriceCategory(price) {
    if (price < 300000) return 'Budget Car 🚗';
    if (price < 800000) return 'Mid-range Car 🚙';
    if (price < 1500000) return 'Premium Car 🚘';
    return 'Luxury Car 🏎️';
}

function getCategoryColor(category) {
    if (category.includes('Budget')) return '#4caf50';
    if (category.includes('Mid-range')) return '#ff9800';
    if (category.includes('Premium')) return '#e91e63';
    return '#9c27b0';
}

function getConfidenceLevel(price) {
    if (price < 500000 || price > 2000000) return 'High confidence 🎯';
    if (price < 1000000) return 'Medium confidence ⚠️';
    return 'Good confidence 👍';
}

function animateNumber(element, start, end, duration) {
    const startTime = performance.now();
    const range = end - start;

    function updateNumber(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const current = start + (range * easeOutQuart(progress));

        element.textContent = `₹${Math.round(current).toLocaleString()}`;

        if (progress < 1) {
            requestAnimationFrame(updateNumber);
        }
    }

    requestAnimationFrame(updateNumber);
}

function easeOutQuart(t) {
    return 1 - Math.pow(1 - t, 4);
}

function showErrorModal(message) {
    errorMessage.textContent = message;
    errorModal.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function hideErrorModal() {
    errorModal.classList.remove('active');
    document.body.style.overflow = '';
}

// Utility function to test API connection
async function testAPIConnection() {
    try {
        const response = await fetch(`${API_BASE_URL}/`);
        const data = await response.json();
        console.log('API Connection Test:', data);
        return true;
    } catch (error) {
        console.error('API Connection Failed:', error);
        return false;
    }
}

// Auto-test API connection on load
testAPIConnection().then(connected => {
    if (!connected) {
        console.warn('API server is not running. Please start the FastAPI server with: uvicorn api_app:app --reload');
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', function (e) {
    // Ctrl/Cmd + Enter to submit form
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (!predictBtn.disabled) {
            predictionForm.dispatchEvent(new Event('submit'));
        }
    }

    // Escape to close modal
    if (e.key === 'Escape') {
        hideErrorModal();
    }
});

// Add some visual feedback for form validation
function validateForm() {
    const requiredFields = predictionForm.querySelectorAll('[required]');
    let isValid = true;

    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            field.classList.add('error');
            isValid = false;
        } else {
            field.classList.remove('error');
        }
    });

    return isValid;
}

// Real-time form validation
predictionForm.addEventListener('input', function (e) {
    if (e.target.hasAttribute('required')) {
        if (e.target.value.trim()) {
            e.target.classList.remove('error');
            e.target.classList.add('success');
        } else {
            e.target.classList.add('error');
            e.target.classList.remove('success');
        }
    }
});

// Add smooth scrolling for internal links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Progressive Web App features
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        // Register service worker for PWA functionality
        console.log('PWA features available');
    });
}

// Handle offline status
window.addEventListener('online', () => {
    console.log('Connection restored');
    testAPIConnection();
});

window.addEventListener('offline', () => {
    console.log('Connection lost');
    showErrorModal('You are currently offline. Please check your internet connection.');
});

// Export functions for testing
window.VehiclePricePredictor = {
    testAPIConnection,
    validateForm,
    displayPredictionResult,
    displayFeatureImportance
};
