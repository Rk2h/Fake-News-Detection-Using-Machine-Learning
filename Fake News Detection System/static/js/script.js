
/* ==========================================================
   Fake News Detection AI
   Clean & Organized script.js
   Part 1 (Lines 1–120)
========================================================== */

document.addEventListener("DOMContentLoaded", function () {

    console.log("Fake News Detection AI Loaded");

    /* ==========================================================
       LOADING SPINNER
    ========================================================== */

    const newsForm = document.getElementById("newsForm");
    const predictButton = document.getElementById("predictButton");
    const loadingSpinner = document.getElementById("loadingSpinner");

    if (newsForm && predictButton && loadingSpinner) {

        newsForm.addEventListener("submit", function () {

            predictButton.disabled = true;

            loadingSpinner.classList.remove("d-none");

            predictButton.innerHTML =
                '<span class="spinner-border spinner-border-sm me-2"></span>Analyzing...';

        });

    }

    /* ==========================================================
       DARK MODE
    ========================================================== */

    const themeToggle = document.getElementById("themeToggle");

    if (themeToggle) {

        if (localStorage.getItem("theme") === "dark") {

            document.body.classList.add("dark-mode");

            themeToggle.innerHTML =
                '<i class="bi bi-sun-fill"></i> Light Mode';

        }

        themeToggle.addEventListener("click", function () {

            document.body.classList.toggle("dark-mode");

            if (document.body.classList.contains("dark-mode")) {

                localStorage.setItem("theme", "dark");

                themeToggle.innerHTML =
                    '<i class="bi bi-sun-fill"></i> Light Mode';

            } else {

                localStorage.setItem("theme", "light");

                themeToggle.innerHTML =
                    '<i class="bi bi-moon-stars-fill"></i> Dark Mode';

            }

        });

    }

    /* ==========================================================
       CONFIDENCE GAUGE
    ========================================================== */

    const circle = document.getElementById("gaugeProgress");
    const confidenceValue = document.getElementById("confidenceValue");

    if (circle && confidenceValue) {

        let confidence = parseFloat(
            confidenceValue.innerText.replace("%", "")
        );

        if (isNaN(confidence)) confidence = 0;

        const radius = 90;
        const circumference = 2 * Math.PI * radius;

        circle.style.strokeDasharray = circumference;
        circle.style.strokeDashoffset = circumference;

        let gaugeColor = "#dc3545";

        if (confidence >= 90) {

            gaugeColor = "#198754";

        } else if (confidence >= 75) {

            gaugeColor = "#20c997";

        } else if (confidence >= 60) {

            gaugeColor = "#ffc107";

        } else if (confidence >= 40) {

            gaugeColor = "#fd7e14";

        }

        circle.style.stroke = gaugeColor;

        confidenceValue.style.color = gaugeColor;

        circle.style.filter =
            `drop-shadow(0 0 10px ${gaugeColor})`;

        setTimeout(function () {

            const offset =
                circumference -
                (confidence / 100) * circumference;

            circle.style.strokeDashoffset = offset;

        }, 300);

    }

    /* ==========================================================
       HISTORY SEARCH
    ========================================================== */

    const historySearch =
        document.getElementById("historySearch");

    if (historySearch) {

        historySearch.addEventListener("keyup", function () {

            const keyword =
                this.value.toLowerCase();

            const rows =
                document.querySelectorAll(".historyRow");

            rows.forEach(function (row) {

                row.style.display =
                    row.innerText.toLowerCase().includes(keyword)
                        ? ""
                        : "none";

            });

        });

    }


    /* ==========================================================
       ANIMATED COUNTERS
    ========================================================== */

    const counters = document.querySelectorAll(".counter");

    counters.forEach(function (counter) {

        const target = parseInt(
            counter.getAttribute("data-target") ||
            counter.innerText
        );

        if (isNaN(target)) return;

        let count = 0;

        const speed = Math.max(target / 80, 1);

        function updateCounter() {

            count += speed;

            if (count < target) {

                counter.innerText = Math.ceil(count);

                requestAnimationFrame(updateCounter);

            } else {

                counter.innerText = target;

            }

        }

        updateCounter();

    });

    /* ==========================================================
       CHART.JS CHECK
    ========================================================== */

    if (typeof Chart === "undefined") {

        console.error("Chart.js is not loaded.");

        return;

    }

    /* ==========================================================
       PREDICTION DISTRIBUTION (DOUGHNUT)
    ========================================================== */

    const predictionChartCanvas =
        document.getElementById("predictionChart");

    if (predictionChartCanvas) {

        new Chart(predictionChartCanvas, {

            type: "doughnut",

            data: {

                labels: [

                    "Real News",

                    "Fake News"

                ],

                datasets: [{

                    data: [

                        realCount,

                        fakeCount

                    ],

                    backgroundColor: [

                        "#198754",

                        "#dc3545"

                    ],

                    borderWidth: 2,

                    hoverOffset: 12

                }]

            },

            options: {

                responsive: true,

                maintainAspectRatio: false,

                cutout: "70%",

                animation: {

                    animateRotate: true,

                    animateScale: true,

                    duration: 1800

                },

                plugins: {

                    legend: {

                        position: "bottom",

                        labels: {

                            padding: 20,

                            font: {

                                size: 14

                            }

                        }

                    },

                    tooltip: {

                        callbacks: {

                            label: function (context) {

                                return (
                                    context.label +
                                    ": " +
                                    context.raw
                                );

                            }

                        }

                    }

                }

            }

        });

    }

    /* ==========================================================
       CHART HELPER
    ========================================================== */

    function createGradient(ctx) {

        const gradient =
            ctx.createLinearGradient(0, 0, 0, 350);

        gradient.addColorStop(
            0,
            "rgba(13,110,253,0.35)"
        );

        gradient.addColorStop(
            1,
            "rgba(13,110,253,0.02)"
        );

        return gradient;

    }


    /* ==========================================================
       CONFIDENCE TREND (LINE CHART)
    ========================================================== */

    const confidenceChartCanvas =
        document.getElementById("confidenceChart");

    if (confidenceChartCanvas) {

        const ctx = confidenceChartCanvas.getContext("2d");

        const gradient = createGradient(ctx);

        new Chart(ctx, {

            type: "line",

            data: {

                labels: predictionLabels,

                datasets: [{

                    label: "Confidence (%)",

                    data: confidenceHistory,

                    borderColor: "#0d6efd",

                    backgroundColor: gradient,

                    fill: true,

                    borderWidth: 3,

                    tension: 0.40,

                    pointRadius: 5,

                    pointHoverRadius: 8,

                    pointBackgroundColor: "#0d6efd",

                    pointBorderColor: "#ffffff",

                    pointBorderWidth: 2

                }]

            },

            options: {

                responsive: true,

                maintainAspectRatio: false,

                animation: {

                    duration: 1800,

                    easing: "easeOutQuart"

                },

                interaction: {

                    intersect: false,

                    mode: "index"

                },

                plugins: {

                    legend: {

                        display: true,

                        position: "top"

                    },

                    tooltip: {

                        callbacks: {

                            label: function(context){

                                return "Confidence: " +
                                       context.parsed.y + "%";

                            }

                        }

                    }

                },

                scales: {

                    y: {

                        beginAtZero: true,

                        max: 100,

                        ticks: {

                            callback: function(value){

                                return value + "%";

                            }

                        },

                        grid: {

                            color: "rgba(0,0,0,0.08)"

                        }

                    },

                    x: {

                        grid: {

                            display: false

                        }

                    }

                }

            }

        });

    }

    /* ==========================================================
       PREDICTION HISTORY (BAR CHART)
    ========================================================== */

    const historyChartCanvas =
        document.getElementById("historyChart");

    if (historyChartCanvas) {

        new Chart(historyChartCanvas, {

            type: "bar",

            data: {

                labels: predictionLabels,

                datasets: [{

                    label: "Confidence",

                    data: confidenceHistory,

                    backgroundColor: "#6f42c1",

                    borderRadius: 10,

                    borderSkipped: false,

                    maxBarThickness: 40

                }]

            },

            options: {

                responsive: true,

                maintainAspectRatio: false,

                animation: {

                    duration: 1800,

                    easing: "easeOutBounce"

                },

                plugins: {

                    legend: {

                        display: false

                    },

                    tooltip: {

                        callbacks: {

                            label: function(context){

                                return context.raw + "%";

                            }

                        }

                    }

                },

                scales: {

                    y: {

                        beginAtZero: true,

                        max: 100

                    }

                }

            }

        });

    }


    /* ==========================================================
       BOOTSTRAP TOOLTIPS
    ========================================================== */

    if (typeof bootstrap !== "undefined") {

        const tooltipTriggerList = [].slice.call(
            document.querySelectorAll('[data-bs-toggle="tooltip"]')
        );

        tooltipTriggerList.forEach(function (tooltipTriggerEl) {

            new bootstrap.Tooltip(tooltipTriggerEl);

        });

    }

    /* ==========================================================
       AUTO HIDE ALERTS
    ========================================================== */

    const alerts = document.querySelectorAll(".alert-dismissible");

    alerts.forEach(function (alert) {

        setTimeout(function () {

            try {

                const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);

                bsAlert.close();

            } catch (e) {

                alert.remove();

            }

        }, 5000);

    });

    /* ==========================================================
       SCROLL TO TOP BUTTON
    ========================================================== */

    const scrollButton = document.getElementById("scrollTopBtn");

    if (scrollButton) {

        window.addEventListener("scroll", function () {

            if (window.scrollY > 300) {

                scrollButton.classList.remove("d-none");

            } else {

                scrollButton.classList.add("d-none");

            }

        });

        scrollButton.addEventListener("click", function () {

            window.scrollTo({

                top: 0,

                behavior: "smooth"

            });

        });

    }

    /* ==========================================================
       FADE-IN ANIMATION
    ========================================================== */

    const fadeElements = document.querySelectorAll(".fade-in");

    if ("IntersectionObserver" in window) {

        const observer = new IntersectionObserver(function (entries) {

            entries.forEach(function (entry) {

                if (entry.isIntersecting) {

                    entry.target.classList.add("show");

                    observer.unobserve(entry.target);

                }

            });

        }, {

            threshold: 0.2

        });

        fadeElements.forEach(function (element) {

            observer.observe(element);

        });

    }

    /* ==========================================================
       AOS INITIALIZATION
    ========================================================== */

    if (typeof AOS !== "undefined") {

        AOS.init({

            once: true,

            duration: 900,

            easing: "ease-in-out"

        });

    }

    /* ==========================================================
       DEBUG INFORMATION
    ========================================================== */

    console.log("====================================");
    console.log(" Fake News Detection AI");
    console.log(" script.js loaded successfully");
    console.log("====================================");

}); // End DOMContentLoaded