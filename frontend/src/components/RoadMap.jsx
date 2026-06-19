import { MapContainer, TileLayer, GeoJSON, Marker, Popup } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix for default marker icons in React
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

let DefaultIcon = L.icon({
    iconUrl: icon,
    shadowUrl: iconShadow,
    iconSize: [25, 41],
    iconAnchor: [12, 41]
});
L.Marker.prototype.options.icon = DefaultIcon;

const RoadMap = ({ geoData }) => {
    if (!geoData) return <div className="glass-panel" style={{ height: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>Awaiting spatial data...</div>;

    const style = (feature) => {
        const score = feature.properties.safety_score;
        let color = "#ff0000"; // Critical
        if (score >= 80) color = "#00ff73"; // Good
        else if (score >= 60) color = "#ffd900"; // Moderate
        
        return {
            color: color,
            weight: 8,
            opacity: 0.8
        };
    };

    const onEachFeature = (feature, layer) => {
        if (feature.properties) {
            layer.bindPopup(`
                <strong>Frame: ${feature.properties.frame_id}</strong><br/>
                Score: ${feature.properties.safety_score}<br/>
                Width: ${feature.properties.road_width_m}m
            `);
        }
    };

    // Calculate center from data
    const firstCoord = geoData.features[0]?.geometry.coordinates[0];
    const center = firstCoord ? [firstCoord[1], firstCoord[0]] : [20.5937, 78.9629]; // Default India

    return (
        <div className="glass-panel" style={{ height: '500px', padding: '0', overflow: 'hidden', border: '1px solid var(--accent-primary)' }}>
            <MapContainer center={center} zoom={15} style={{ height: '100%', width: '100%' }}>
                <TileLayer
                    url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
                />
                <GeoJSON 
                    data={geoData} 
                    style={style} 
                    onEachFeature={onEachFeature} 
                />
            </MapContainer>
        </div>
    );
};

export default RoadMap;
