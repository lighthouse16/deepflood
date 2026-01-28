// Map with enhanced styling
const map = L.map('map').setView([17.4005, 106.4100], 9);
L.tileLayer('https://api.maptiler.com/maps/basic-v2/{z}/{x}/{y}.png?key=Cr3tHNjg7FKkEsm1SvyR', {
  attribution: '&copy; <a href="https://www.maptiler.com/">MapTiler</a> & OpenStreetMap contributors',
  tileSize: 512,
  zoomOffset: -1
}).addTo(map);

// Custom marker with enhanced popup
const customIcon = L.divIcon({
  className: 'custom-marker',
  html: '<div style="background: linear-gradient(135deg, #667eea, #764ba2); width: 30px; height: 30px; border-radius: 50%; border: 3px solid white; box-shadow: 0 2px 10px rgba(0,0,0,0.3); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">🌊</div>',
  iconSize: [30, 30],
  iconAnchor: [15, 15]
});

L.marker([17.222917, 106.456250], {icon: customIcon})
  .addTo(map)
  .bindPopup(`
    <div style="text-align: center; padding: 10px;">
      <h3 style="color: #1e3c72; margin: 0 0 10px 0;">🌊 Long Dai River</h3>
      <p style="margin: 5px 0;"><strong>Streamflow:</strong> 7.7 m³/s</p>
      <p style="margin: 5px 0;"><strong>Risk:</strong> <span style="color: #48dbfb; font-weight: bold;">Low</span></p>
      <div style="background: linear-gradient(135deg, #667eea, #764ba2); height: 4px; border-radius: 2px; margin-top: 10px;"></div>
    </div>
  `)
  .openPopup();

// Enhanced Chart with better styling
const forecastCtx = document.getElementById('forecastChart').getContext('2d');
new Chart(forecastCtx, {
  type: 'bar',
  data: {
    labels: ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
    datasets: [{
      label: 'Streamflow',
      data: [12.7, 10.6, 6.8, 5.1, 3, 1.5, 0.2],
      backgroundColor: [
        'rgba(102, 126, 234, 0.8)',
        'rgba(118, 75, 162, 0.8)',
        'rgba(72, 219, 251, 0.8)',
        'rgba(254, 202, 87, 0.8)',
        'rgba(255, 159, 243, 0.8)',
        'rgba(102, 126, 234, 0.6)',
        'rgba(118, 75, 162, 0.6)'
      ],
      borderColor: [
        'rgba(102, 126, 234, 1)',
        'rgba(118, 75, 162, 1)',
        'rgba(72, 219, 251, 1)',
        'rgba(254, 202, 87, 1)',
        'rgba(255, 159, 243, 1)',
        'rgba(102, 126, 234, 1)',
        'rgba(118, 75, 162, 1)'
      ],
      borderWidth: 2,
      borderRadius: 8,
      borderSkipped: false,
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: '#1e3c72',
          font: {
            family: 'Be Vietnam Pro',
            size: 12,
            weight: '600'
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: { 
          display: true, 
          text: 'm³/s',
          color: '#1e3c72',
          font: {
            family: 'Be Vietnam Pro',
            size: 14,
            weight: '600'
          }
        },
        grid: {
          color: 'rgba(102, 126, 234, 0.1)'
        },
        ticks: {
          color: '#1e3c72',
          font: {
            family: 'Be Vietnam Pro',
            size: 12
          }
        }
      },
      x: {
        grid: {
          color: 'rgba(102, 126, 234, 0.1)'
        },
        ticks: {
          color: '#1e3c72',
          font: {
            family: 'Be Vietnam Pro',
            size: 12,
            weight: '600'
          }
        }
      }
    }
  }
});

// Add some interactive effects
document.addEventListener('DOMContentLoaded', function() {
  // Add hover effect for weather items
  const weatherItems = document.querySelectorAll('.weather-item');
  weatherItems.forEach(item => {
    item.addEventListener('mouseenter', function() {
      this.style.transform = 'scale(1.05)';
    });
    item.addEventListener('mouseleave', function() {
      this.style.transform = 'scale(1)';
    });
  });

  // Add click effect for forecast list items
  const forecastItems = document.querySelectorAll('.forecast-list li');
  forecastItems.forEach(item => {
    item.addEventListener('click', function() {
      this.style.background = 'rgba(102, 126, 234, 0.1)';
      setTimeout(() => {
        this.style.background = '';
      }, 300);
    });
  });
});

// Predicted streamflow chart from predicted_streamflow_long_dai.csv
function parseCSV(text) {
  const lines = text.trim().split('\n');
  const headers = lines[0].split(',');
  const data = lines.slice(1).map(line => {
    const values = line.split(',');
    return {
      date: values[0],
      predicted_streamflow_cms: parseFloat(values[1])
    };
  });
  return data;
}

function createStreamflowChart(data) {
  const labels = data.map(row => row.date);
  const values = data.map(row => row.predicted_streamflow_cms);

  const ctx = document.getElementById('streamflowChart');
  if (!ctx) {
    console.error('streamflowChart canvas not found');
    return;
  }

  new Chart(ctx.getContext('2d'), {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Predicted Streamflow (m³/s)',
        data: values,
        borderColor: 'rgba(102, 126, 234, 1)',
        backgroundColor: 'rgba(102, 126, 234, 0.1)',
        pointRadius: 0,
        fill: true,
        tension: 0.2
      },
      {
        label: 'Low Alert (300 m³/s)',
        data: Array(labels.length).fill(300),
        borderColor: 'rgba(255, 193, 7, 0.8)',
        backgroundColor: 'transparent',
        borderWidth: 2,
        borderDash: [5, 5],
        pointRadius: 0,
        fill: false
      },
      {
        label: 'Medium Alert (500 m³/s)',
        data: Array(labels.length).fill(500),
        borderColor: 'rgba(255, 152, 0, 0.8)',
        backgroundColor: 'transparent',
        borderWidth: 2,
        borderDash: [5, 5],
        pointRadius: 0,
        fill: false
      },
      {
        label: 'High Alert (900 m³/s)',
        data: Array(labels.length).fill(900),
        borderColor: 'rgba(244, 67, 54, 0.8)',
        backgroundColor: 'transparent',
        borderWidth: 2,
        borderDash: [5, 5],
        pointRadius: 0,
        fill: false
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: {
            color: '#1e3c72',
            font: { size: 14, weight: 'bold' }
          }
        }
      },
      scales: {
        x: {
          display: true,
          title: {
            display: true,
            text: 'Date',
            color: '#1e3c72',
            font: { size: 14, weight: 'bold' }
          },
          ticks: {
            color: '#1e3c72',
            maxTicksLimit: 12,
            autoSkip: true
          }
        },
        y: {
          display: true,
          title: {
            display: true,
            text: 'Streamflow (m³/s)',
            color: '#1e3c72',
            font: { size: 14, weight: 'bold' }
          },
          ticks: {
            color: '#1e3c72'
          }
        }
      }
    }
  });
}

// Load CSV data
fetch('data/predicted_streamflow_long_dai.csv')
  .then(response => {
    if (!response.ok) {
      throw new Error('CSV fetch failed: ' + response.statusText);
    }
    return response.text();
  })
  .then(csvText => {
    const data = parseCSV(csvText);
    if (data.length > 0) {
      createStreamflowChart(data);
    } else {
      console.error('No data parsed from CSV');
    }
  })
  .catch(error => {
    console.error('Error loading streamflow data:', error);
    // Fallback with sample data
    const sampleData = [
      {date: '2024-08-05', predicted_streamflow_cms: 1.93},
      {date: '2024-08-06', predicted_streamflow_cms: 2.88},
      {date: '2024-08-07', predicted_streamflow_cms: 1.92},
      {date: '2024-08-08', predicted_streamflow_cms: 1.67},
      {date: '2024-08-09', predicted_streamflow_cms: 3.73},
      {date: '2024-08-10', predicted_streamflow_cms: 2.93},
      {date: '2024-08-11', predicted_streamflow_cms: 3.04},
      {date: '2024-08-12', predicted_streamflow_cms: 2.66},
      {date: '2024-08-13', predicted_streamflow_cms: 4.03},
      {date: '2024-08-14', predicted_streamflow_cms: 4.02}
    ];
    createStreamflowChart(sampleData);
  });
