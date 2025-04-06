document.addEventListener('DOMContentLoaded', function() {
    // Initialize map centered at a default location (India)
    const map = L.map('map').setView([20.5937, 78.9629], 5);
    
    // Add OpenStreetMap tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);
    
    // Variables to store map data
    let potholeMarkers = [];
    let currentRoute = null;
    let searchTimeout = null;
    let routeMarkers = [];
    
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
            
            // Reverse geocode to get user's address
            reverseGeocode(userLat, userLng).then(address => {
                if (address) {
                    document.getElementById('start').value = address;
                    document.getElementById('start').dataset.lat = userLat;
                    document.getElementById('start').dataset.lng = userLng;
                }
            });
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
    
    // Enhanced geocoding function for location search
    const geocodeLocation = async (query) => {
        if (!query || query.length < 3) return [];
        
        try {
            // Use Nominatim OpenStreetMap geocoding (free, no API key needed)
            const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=5`;
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error(`Geocoding error: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Transform the data to a more usable format
            return data.map(item => ({
                lat: parseFloat(item.lat),
                lng: parseFloat(item.lon),
                display_name: item.display_name,
                name: item.name || item.display_name.split(',')[0],
                type: item.type,
                importance: item.importance
            }));
        } catch (error) {
            console.error("Geocoding error:", error);
            return [];
        }
    };
    
    // Reverse geocode to get address from coordinates
    const reverseGeocode = async (lat, lng) => {
        try {
            const url = `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}`;
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error(`Reverse geocoding error: ${response.status}`);
            }
            
            const data = await response.json();
            return data.display_name;
        } catch (error) {
            console.error("Reverse geocoding error:", error);
            return null;
        }
    };
    
    // Function to create and display search results
    function displaySearchResults(results, containerId) {
        const container = document.getElementById(containerId);
        container.innerHTML = '';
        
        if (results.length === 0) {
            const noResults = document.createElement('div');
            noResults.className = 'search-result no-results';
            noResults.textContent = 'No locations found';
            container.appendChild(noResults);
            return;
        }
        
        results.forEach(result => {
            const resultItem = document.createElement('div');
            resultItem.className = 'search-result';
            
            const name = document.createElement('div');
            name.className = 'result-name';
            name.textContent = result.name;
            
            const address = document.createElement('div');
            address.className = 'result-address';
            address.textContent = result.display_name;
            
            resultItem.appendChild(name);
            resultItem.appendChild(address);
            
            // Add click handler
            resultItem.addEventListener('click', () => {
                // Fill the input with selected location
                if (containerId === 'start-results') {
                    document.getElementById('start').value = result.display_name;
                    document.getElementById('start').dataset.lat = result.lat;
                    document.getElementById('start').dataset.lng = result.lng;
                    document.getElementById('start-results').innerHTML = '';
                } else {
                    document.getElementById('destination').value = result.display_name;
                    document.getElementById('destination').dataset.lat = result.lat;
                    document.getElementById('destination').dataset.lng = result.lng;
                    document.getElementById('destination-results').innerHTML = '';
                }
            });
            
            container.appendChild(resultItem);
        });
    }
    
    // Setup input event listeners for both inputs with debouncing
    document.getElementById('start').addEventListener('input', function(e) {
        const resultsContainer = document.getElementById('start-results');
        resultsContainer.innerHTML = '';
        
        clearTimeout(searchTimeout);
        
        if (e.target.value.length < 3) return;
        
        searchTimeout = setTimeout(async () => {
            const results = await geocodeLocation(e.target.value);
            displaySearchResults(results, 'start-results');
        }, 500);
    });
    
    document.getElementById('destination').addEventListener('input', function(e) {
        const resultsContainer = document.getElementById('destination-results');
        resultsContainer.innerHTML = '';
        
        clearTimeout(searchTimeout);
        
        if (e.target.value.length < 3) return;
        
        searchTimeout = setTimeout(async () => {
            const results = await geocodeLocation(e.target.value);
            displaySearchResults(results, 'destination-results');
        }, 500);
    });
    
    // Handle click outside search results to close them
    document.addEventListener('click', function(e) {
        if (!e.target.closest('.search-container')) {
            document.getElementById('start-results').innerHTML = '';
            document.getElementById('destination-results').innerHTML = '';
        }
    });
    
    // Get route from OSRM API
    async function getRoute(startLat, startLng, endLat, endLng) {
        try {
            // Using OSRM's free API for routing
            const url = `https://router.project-osrm.org/route/v1/driving/${startLng},${startLat};${endLng},${endLat}?steps=true&geometries=geojson&overview=full&annotations=true`;
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error(`Routing error: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error("Routing error:", error);
            throw error;
        }
    }
    
    // Display turn-by-turn directions in the UI
    function displayDirections(steps) {
        const container = document.getElementById('directions-container');
        container.innerHTML = '';
        
        const directionsList = document.createElement('ol');
        directionsList.className = 'directions-list';
        
        steps.forEach((step, index) => {
            const listItem = document.createElement('li');
            listItem.className = 'direction-step';
            
            // Format the instruction
            const instruction = document.createElement('div');
            instruction.className = 'direction-instruction';
            instruction.textContent = step.maneuver.type + ' onto ' + (step.name || 'unnamed road');
            
            // Add distance if available
            if (step.distance) {
                const distance = document.createElement('span');
                distance.className = 'direction-distance';
                distance.textContent = `${(step.distance / 1000).toFixed(1)} km`;
                instruction.appendChild(distance);
            }
            
            listItem.appendChild(instruction);
            directionsList.appendChild(listItem);
            
            // Add markers to the map for each turn (only for major turns to avoid clutter)
            if (step.maneuver.type !== 'depart' && step.maneuver.type !== 'arrive') {
                const markerIcon = L.divIcon({
                    className: 'turn-marker',
                    html: `<div class="turn-marker-number">${index + 1}</div>`,
                    iconSize: [24, 24]
                });
                
                const marker = L.marker([step.maneuver.location[1], step.maneuver.location[0]], {
                    icon: markerIcon
                }).addTo(map);
                
                marker.bindPopup(step.maneuver.type + ' onto ' + (step.name || 'unnamed road'));
                routeMarkers.push(marker);
            }
        });
        
        container.appendChild(directionsList);
    }

    // Clear previous route and markers
    function clearRoute() {
        if (currentRoute) {
            map.removeLayer(currentRoute);
            currentRoute = null;
        }
        
        routeMarkers.forEach(marker => map.removeLayer(marker));
        routeMarkers = [];
    }
    
    // Route finding button handler
    document.getElementById('route-btn').addEventListener('click', async function() {
        const startInput = document.getElementById('start');
        const destInput = document.getElementById('destination');
        const statusEl = document.getElementById('route-status');
        
        // Reset status
        statusEl.innerHTML = 'Finding route...';
        
        // Get inputs from form
        const startValue = startInput.value;
        const destValue = destInput.value;
        
        if (!startValue || !destValue) {
            statusEl.innerHTML = '<div class="route-status-error">Please enter both starting point and destination</div>';
            return;
        }
        
        try {
            // Clear any previous route
            clearRoute();
            
            // Check if coordinates are stored in data attributes
            let startLoc, destLoc;
            
            if (startInput.dataset.lat && startInput.dataset.lng) {
                startLoc = {
                    lat: parseFloat(startInput.dataset.lat),
                    lng: parseFloat(startInput.dataset.lng),
                    name: startValue.split(',')[0]
                };
            } else {
                // Geocode if not available
                const startResults = await geocodeLocation(startValue);
                if (startResults.length === 0) {
                    statusEl.innerHTML = '<div class="route-status-error">Could not find starting location. Please try a different search term.</div>';
                    return;
                }
                startLoc = startResults[0];
            }
            
            if (destInput.dataset.lat && destInput.dataset.lng) {
                destLoc = {
                    lat: parseFloat(destInput.dataset.lat),
                    lng: parseFloat(destInput.dataset.lng),
                    name: destValue.split(',')[0]
                };
            } else {
                // Geocode if not available
                const destResults = await geocodeLocation(destValue);
                if (destResults.length === 0) {
                    statusEl.innerHTML = '<div class="route-status-error">Could not find destination. Please try a different search term.</div>';
                    return;
                }
                destLoc = destResults[0];
            }
            
            // Get route from API
            const routeData = await getRoute(startLoc.lat, startLoc.lng, destLoc.lat, destLoc.lng);
            
            if (!routeData.routes || routeData.routes.length === 0) {
                statusEl.innerHTML = '<div class="route-status-error">No route found between these locations.</div>';
                return;
            }
            
            const route = routeData.routes[0];
            
            // Draw route on map using GeoJSON
            currentRoute = L.geoJSON(route.geometry, {
                style: {
                    color: 'blue',
                    weight: 5,
                    opacity: 0.7
                }
            }).addTo(map);
            
            // Fit map to route bounds
            map.fitBounds(currentRoute.getBounds());
            
            // Display turn-by-turn directions
            if (route.legs && route.legs.length > 0 && route.legs[0].steps) {
                displayDirections(route.legs[0].steps);
            }
            
            // Calculate route bounds to query for potholes
            const bounds = currentRoute.getBounds();
            const bufferDeg = 0.02;  // About 2km buffer
            
            // Query for potholes along route
            const response = await fetch(`/api/route?start_lat=${bounds.getSouth() - bufferDeg}&start_lng=${bounds.getWest() - bufferDeg}&end_lat=${bounds.getNorth() + bufferDeg}&end_lng=${bounds.getEast() + bufferDeg}`);
            const data = await response.json();
            
            // Update pothole count display
            document.getElementById('pothole-count').textContent = `Potholes on route: ${data.pothole_count}`;
            
            // Update status with formatted information
            statusEl.innerHTML = `
                <div class="route-status-success">
                    <div class="route-summary">
                        <strong>Route found:</strong> ${startLoc.name || startValue.split(',')[0]} to ${destLoc.name || destValue.split(',')[0]}
                    </div>
                    <div class="route-details">
                        <span class="detail-item">Distance: ${(route.distance / 1000).toFixed(1)} km</span>
                        <span class="detail-item">Duration: ${Math.round(route.duration / 60)} min</span>
                        <span class="detail-item">Potholes: ${data.pothole_count}</span>
                    </div>
                </div>
            `;
            
        } catch (error) {
            console.error('Error in route processing:', error);
            statusEl.innerHTML = '<div class="route-status-error">An error occurred while processing your route request.</div>';
        }
    });
    
    // Set up periodic refresh of pothole data (every 30 seconds)
    setInterval(fetchPotholes, 30000);
});
