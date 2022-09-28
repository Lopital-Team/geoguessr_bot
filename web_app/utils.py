import folium

def make_html(longitude=None, latitude=None, longitude_true=None, latitude_true=None):
    zones = None
    if (not longitude):
        pred_map = Visualizer(zones=zones)
    else:
        if(not longitude_true):
            pred_map = Visualizer(pred_loc=[latitude, longitude], zones=zones)
        else:
            pred_map = Visualizer(pred_loc=[latitude, longitude], true_loc=[latitude_true, longitude_true], zones=zones)
    pred_map.add_elements()
    pred_map.save()
    
class Visualizer():
    '''
    Defines map and map elements such as points, grids and connecting lines.
    '''
    def __init__(self,true_loc = None,pred_loc = None,zones = None):
        self.map_ = folium.Map(location = [44.65,16.06], zoom_start = 7)
        self.zones = zones
        self.true_loc = true_loc
        self.pred_loc = pred_loc
        
        
    def add_elements(self):
        self.create_grid()
        self.add_true_loc()
        self.add_pred_loc()
        self.add_line()
        return
            
    def add_line(self):
        if not self.true_loc:
            return
        if not self.pred_loc:
            return
        folium.PolyLine([self.true_loc,self.pred_loc], color="red", weight=2.5, opacity=1).add_to(self.map_)
        
        return
            
    def add_pred_loc(self):
        if not self.pred_loc:
            return
        folium.Marker(location = self.pred_loc, popup="Predicted location", icon=folium.Icon(color="red")).add_to(self.map_)
        
        return
    
    def add_true_loc(self):
        if not self.true_loc:
            return
        folium.Marker(location = self.true_loc, popup="True location", icon=folium.Icon(color="blue")).add_to(self.map_)
        
        return
    
    def create_grid(self):
        if not self.zones:
            return
        geo_json = self.create_gjson(self.zones)
        style = self.get_style()
        folium.GeoJson(data =geo_json, style_function = lambda x: style).add_to(self.map_)

        folium.LayerControl().add_to(self.map_)
        return
    
    def get_style(self):
        border_style = {
            'color': 'green',
            'weight': 2,
            'fillColor': 'blue',
            'fillOpacity': 0.2
            }    
        return border_style
    def create_gjson(self,zones):
        geo_json = {
            "type" : "FeatureCollection",
            "features":[
        {
            "type":"Feature",
            "geometry":{
                "type" : "Polygon",
                "coordinates" : [zone]
            }
            
        } for zone in zones]
        }
        return geo_json
    def save(self):
        return self.map_.save("./templates/new.html")
    

        
