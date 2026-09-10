

import pandas as pd
import math


class RouteDataProcessor:

    def __init__(self, route_csv, duration_days, times, truck_consumption):
        self.routes_df = pd.read_csv(route_csv)
        self.duration_days = duration_days
        self.times = times
        self.truck_consumption = truck_consumption
        self.process_routes()

    # ============================================================
    # Convert HH:MM to decimal hours
    # ============================================================
    def time_to_hours(self, time):
        hours, minutes = str(time).split(":")
        return int(hours) + int(minutes) / 60

    # ============================================================
    # Process Route Data
    # ============================================================
    def process_routes(self):
        # --------------------------------------------------------
        # Convert times to hourly timesteps
        # --------------------------------------------------------

        self.routes_df["departure_hour"] = (self.routes_df["Pick Up Time"].apply(self.time_to_hours).apply(math.floor))
        self.routes_df["return_hour"] = (self.routes_df["Drop Off Time"].apply(self.time_to_hours).apply(math.ceil))

        # --------------------------------------------------------
        # Repeat routes for each day
        # --------------------------------------------------------
        expanded_routes = []
        route_no = 1

        for day in range(self.duration_days):
            for _, row in self.routes_df.iterrows():
                route = row.copy()
                route["route_no"] = route_no
                route["year"] = day // 365
                route["day"] = day
                route["departure_t"] = (row["departure_hour"] + day * 24)
                route["return_t"] = (row["return_hour"] + day * 24)

                expanded_routes.append(route)
                route_no += 1

        self.routes_df = pd.DataFrame(expanded_routes)
        self.route_year = dict(zip(self.routes_df["route_no"].astype(int), self.routes_df["year"].astype(int)))

        # --------------------------------------------------------
        # Routes and Truck Assignment
        # --------------------------------------------------------

        self.routes = (self.routes_df["route_no"].astype(int).tolist())
        self.trucks = sorted(self.routes_df["Truck No"].astype(int).unique().tolist())
        self.route_truck = dict(zip(self.routes_df["route_no"].astype(int),self.routes_df["Truck No"].astype(int)))


        # --------------------------------------------------------
        # Route departure and return times
        # --------------------------------------------------------

        self.route_departure = dict(zip(self.routes_df["route_no"].astype(int),self.routes_df["departure_t"].astype(int)))
        self.route_return = dict(zip(self.routes_df["route_no"].astype(int),self.routes_df["return_t"].astype(int)))

        # --------------------------------------------------------
        # Route energy requirement
        # --------------------------------------------------------
        
        self.route_energy = {r: (self.route_return[r] - self.route_departure[r]) * self.truck_consumption for r in self.routes}
        
        # --------------------------------------------------------
        # Routes and Trucks active at each timestep
        # --------------------------------------------------------

        self.active_routes_at = {t: [] for t in self.times}

        for r in self.routes:
            for t in range(self.route_departure[r],self.route_return[r]):
                if t in self.active_routes_at:
                    self.active_routes_at[t].append(r)

        self.active_trucks_at = {t: sorted({self.route_truck[r] for r in self.active_routes_at[t]}) for t in self.times}

        # --------------------------------------------------------
        # Overlapping routes
        # --------------------------------------------------------

        self.overlapping_routes = []

        for day in range(self.duration_days):
            day_routes = (self.routes_df[self.routes_df["day"] == day]["route_no"].astype(int).tolist())
            
            for i in range(len(day_routes)):
                for j in range(i + 1, len(day_routes)):
                    r1 = day_routes[i]
                    r2 = day_routes[j]
                    if (self.route_departure[r1] < self.route_return[r2] and self.route_departure[r2] < self.route_return[r1]):
                        self.overlapping_routes.append((r1, r2))

        # --------------------------------------------------------
        # Routes Departing at
        # --------------------------------------------------------
        
        self.routes_departing_at = {t: [] for t in self.times}
        
        for r in self.routes:
            if self.route_departure[r] in self.routes_departing_at:
                self.routes_departing_at[self.route_departure[r]].append(r)

        # --------------------------------------------------------
        # Route pre-departure energy requirement
        # --------------------------------------------------------

        self.route_pre_departure = {r: max(self.times[0], self.route_departure[r] - 1) for r in self.routes}

