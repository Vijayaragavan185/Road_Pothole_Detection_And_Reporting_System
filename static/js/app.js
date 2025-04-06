document.addEventListener('DOMContentLoaded', function() {
    // Initialize map centered at a default location (India)
    const map = L.map('map').setView([20.5937, 78.9629], 5);
    
    // Add OpenStreetMap tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);
    
    // Variables to store map data
    let potholeMarkers = [];
    let currentRoute = null;
    
    // Get user's location if available
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(function(position) {
            const userLat = position.coords.latitude;
            const userLng = position.coords.longitude;
            
            // Center map on user's location
            map.setView([userLat, userLng], 13);
            
            // Add marker for user's location
            L.marker([userLat, userLng]).addTo(map)
                .bindPopup('Your Location')
                .openPopup();
            
            // Load potholes near user's location
            fetchPotholes();
        }, function() {
            // If geolocation fails, just fetch all potholes
            fetchPotholes();
        });
    } else {
        // If geolocation not supported, just fetch all potholes
        fetchPotholes();
    }
    
    // Fetch potholes from API
    function fetchPotholes() {
        fetch('/api/potholes')
            .then(response => response.json())
            .then(data => {
                // Remove existing markers
                potholeMarkers.forEach(marker => map.removeLayer(marker));
                potholeMarkers = [];
                
                // Update count
                document.getElementById('pothole-count').textContent = `Potholes on route: ${data.length}`;
                
                // Add markers for each pothole
                data.forEach(pothole => {
                    const marker = L.circleMarker([pothole.lat, pothole.lng], {
                        radius: 8,
                        fillColor: getSeverityColor(pothole.severity),
                        color: '#000',
                        weight: 1,
                        opacity: 1,
                        fillOpacity: 0.8
                    }).addTo(map);
                    
                    marker.bindPopup(`
                        <strong>Pothole</strong><br>
                        Severity: ${(pothole.severity * 10).toFixed(1)}/10<br>
                        Detected: ${new Date(pothole.timestamp).toLocaleString()}
                    `);
                    
                    potholeMarkers.push(marker);
                });
                
                // Fit map to show all markers if there are any
                if (potholeMarkers.length > 0) {
                    const group = new L.featureGroup(potholeMarkers);
                    map.fitBounds(group.getBounds());
                }
            })
            .catch(error => console.error('Error fetching potholes:', error));
    }
    
    // Get color based on severity
    function getSeverityColor(severity) {
        if (severity > 0.8) return '#ff0000'; // Severe
        if (severity > 0.6) return '#ff8800'; // Moderate
        return '#ffcc00'; // Mild
    }
    
    // Initialize geocoder for location search
    const geocodeLocation = async (location) => {
        // Use Nominatim OpenStreetMap geocoding (free, no API key needed)
        const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(location)}`;
        const response = await fetch(url);
        const data = await response.json();
        
        if (data && data.length > 0) {
            return {
                lat: parseFloat(data[0].lat),
                lng: parseFloat(data[0].lon),
                display_name: data[0].display_name
            };
        }
        return null;
    };
    
    // Route finding functionality
    document.getElementById('route-btn').addEventListener('click', async function() {
        const start = document.getElementById('start').value;
        const destination = document.getElementById('destination').value;
        const statusEl = document.getElementById('route-status');
        
        if (!start || !destination) {
            alert('Please enter both starting point and destination');
            return;
        }
        
        statusEl.innerHTML = 'Finding route...';
        
        try {
            // Geocode both locations
            const startLoc = await geocodeLocation(start);
            const destLoc = await geocodeLocation(destination);
            
            if (!startLoc || !destLoc) {
                statusEl.innerHTML = 'Could not find one or both locations. Please try different search terms.';
                return;
            }
            
            // Clear previous route
            if (currentRoute) {
                map.removeLayer(currentRoute);
            }
            
            // Draw route on map
            currentRoute = L.polyline([
                [startLoc.lat, startLoc.lng],
                [destLoc.lat, destLoc.lng]
            ], {color: 'blue', weight: 4}).addTo(map);
            
            // Fit map to route
            map.fitBounds(currentRoute.getBounds());
            
            // Query for potholes along route
            const response = await fetch(`/api/route?start_lat=${startLoc.lat}&start_lng=${startLoc.lng}&end_lat=${destLoc.lat}&end_lng=${destLoc.lng}`);
            const data = await response.json();
            
            // Update pothole count display
            document.getElementById('pothole-count').textContent = `Potholes on route: ${data.pothole_count}`;
            
            // Update status
            statusEl.innerHTML = `Route found from ${startLoc.display_name.split(',')[0]} to ${destLoc.display_name.split(',')[0]}. ${data.pothole_count} potholes detected along this ${data.distance_km.toFixed(1)} km route.`;
            
        } catch (error) {
            console.error('Error finding route:', error);
            statusEl.innerHTML = 'Error finding route. Please try again.';
        }
    });
    
    // Add suggestions as user types
    document.getElementById('start').addEventListener('input', async function(e) {
        if (e.target.value.length > 3) {
            const results = await geocodeLocation(e.target.value);
            const datalist = document.getElementById('suggestions-start');
            datalist.innerHTML = '';
            
            if (results) {
                const option = document.createElement('option');
                option.value = results.display_name;
                datalist.appendChild(option);
            }
        }
    });
    
    document.getElementById('destination').addEventListener('input', async function(e) {
        if (e.target.value.length > 3) {
            const results = await geocodeLocation(e.target.value);
            const datalist = document.getElementById('suggestions-end');
            datalist.innerHTML = '';
            
            if (results) {
                const option = document.createElement('option');
                option.value = results.display_name;
                datalist.appendChild(option);
            }
        }
    });
    
    // Set up periodic refresh of pothole data (every 30 seconds)
    setInterval(fetchPotholes, 30000);
});
