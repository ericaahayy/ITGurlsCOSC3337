import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from shapely.geometry import Polygon, MultiPolygon

dataset1 = "Solar_flare_RHESSI_2004_05.csv"
df = pd.read_csv(dataset1)


df['dt.start'] = pd.to_datetime(df['dt.start'])
start_date = pd.to_datetime('2004-01-01')
end_date = start_date + pd.DateOffset(months=4)
df = df[(df['dt.start'] >= start_date) & (df['dt.start'] <= end_date)]


flare_events = df[['total.counts', 'x.pos.asec', 'y.pos.asec']]


query_points = {
    'Q1': {'coordinates': (0, 0), 'range': ([-201, 201], [-335, 335])},
    'Q2': {'coordinates': (-805, 0), 'range': ([-1007, -605], [-335, 335])},
    'Q3': {'coordinates': (805, 0), 'range': ([605, 1005], [-355, 335])},
    'Q4': {'coordinates': (-403, 674), 'range': ([-605, -201], [335, 1012])},
    'Q5': {'coordinates': (403, 674), 'range': ([201, 605], [335, 1012])},
    'Q6': {'coordinates': (-403, -667), 'range': ([-605, -201], [-335, -998])},
    'Q7': {'coordinates': (403, -667), 'range': ([201, 605], [-335, -998])},
}


sigma = 250.0


# create grid for the density map
x_min, x_max = -1200, 1200 
y_min, y_max = -1200, 1200  
step = 10      

xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))


grid_points = np.vstack([xx.ravel(), yy.ravel()]).T

# create KDE model 
kde = KernelDensity(bandwidth=sigma, kernel='gaussian')
kde.fit(flare_events[['x.pos.asec', 'y.pos.asec']])  

# calculate the log-densities
log_densities = kde.score_samples(grid_points)

density_map = np.exp(log_densities).reshape(xx.shape)

d1 = 7e-7
d2 = 3e-7

x_scale_factor = (x_max - x_min) / (density_map.shape[1] - 1)
y_scale_factor = (y_max - y_min) / (density_map.shape[0]- 1)

# create MultiPolygons from the density map based on a threshold
def create_multi_polygons(density_map, threshold):
    polygons = []
    
    contour_levels = [threshold]
    
    scaled_contours = plt.contour(
        np.linspace(x_min, x_max, density_map.shape[1]),
        np.linspace(y_min, y_max, density_map.shape[0]),
        density_map,
        levels=contour_levels
    )
    for collection in scaled_contours.collections:
        for path in collection.get_paths():
            for polygon in path.to_polygons():
                polygons.append(Polygon(polygon))
    return MultiPolygon(polygons)

plt.imshow(density_map, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='viridis')
plt.colorbar(label='Density')
plt.scatter([p['coordinates'][0] for p in query_points.values()], [p['coordinates'][1] for p in query_points.values()], color='red', marker='x', label='Query Points')

# scaled contour lines
multi_polygons_d1 = create_multi_polygons(density_map, d1)
multi_polygons_d2 = create_multi_polygons(density_map, d2)

# filter out small hotspots: density above d1
def filter_small_hotspots(multi_polygons, min_area):
    filtered_polygons = []
    for polygon in multi_polygons.geoms:  
        if polygon.area >= min_area:
            filtered_polygons.append(polygon)
    return MultiPolygon(filtered_polygons)

# set the minimum area for a hotspot
min_area_d1 = 1000 


filtered_multi_polygons_d1 = filter_small_hotspots(multi_polygons_d1, min_area_d1)

#create plot for both MultiPolygons
plt.figure(figsize=(8, 6))

# base data
plt.scatter(flare_events['x.pos.asec'], flare_events['y.pos.asec'], s=1, c='lightgray', alpha=0.6, label='Base Data')

# MultiPolygons for d2
for polygon in multi_polygons_d2.geoms:
    x, y = polygon.exterior.xy
    plt.fill(x, y, color='red', alpha=0.6, label='Hotspots (d2)')

# MultiPolygons for d1 
for polygon in filtered_multi_polygons_d1.geoms:
    x, y = polygon.exterior.xy
    plt.fill(x, y, color='blue', alpha=0.6, label='Hotspots (d1)')

plt.xlabel('X (arcseconds)')
plt.ylabel('Y (arcseconds)')
plt.title('Hotspot Density Map for d1 and d2')
plt.legend()

plt.tight_layout()
plt.show()