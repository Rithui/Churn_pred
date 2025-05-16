// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Add animation to the main card
    const mainCard = document.querySelector('.main-card');
    if (mainCard) {
        setTimeout(() => {
            mainCard.style.opacity = '1';
        }, 100);
    }
    
    // Add character counter to textarea
    const textarea = document.getElementById('email_text');
    if (textarea) {
        textarea.addEventListener('input', function() {
            // You could add a character counter here if needed
            if (this.value.length > 0) {
                document.querySelector('.btn-primary').classList.add('active');
            } else {
                document.querySelector('.btn-primary').classList.remove('active');
            }
        });
    }
    
    // Add animation to result container
    const resultContainer = document.querySelector('.result-container');
    if (resultContainer) {
        setTimeout(() => {
            resultContainer.classList.add('show');
        }, 300);
    }
});