"""Process delivery-route schedules for the DairyDrive Electric optimisation model.

Loads a daily route CSV, discretises departure and return times to hourly
boundaries, and repeats the route schedule over the requested number of days.
Produces route-to-truck mappings, diesel and electric energy requirements,
active-route timesteps, overlapping-route pairs and pre-departure indices.

The resulting attributes are consumed by main.py and optimiser.py. Route
energy is estimated using the conversion factor defined in process_routes().
"""

import pandas as pd
import math


class RouteDataProcessor:

    """Convert a delivery-route CSV into indexed optimisation inputs.
    
    Parameters
    ----------
    route_csv : str or path-like
        Path to the route schedule containing Truck No, Pick Up Time,
        Drop Off Time and Diesel Used columns.
    duration_days : int
        Number of days over which to repeat the input schedule.
    times : iterable of int
        Optimisation timestep indices used to build active-route mappings.
    
    Attributes
    ----------
    routes, trucks : list of int
        Expanded route identifiers and unique truck identifiers.
    route_truck, route_departure, route_return : dict
        Route assignment and integer-hour timing mappings.
    route_diesel_used, route_energy : dict
        Route fuel demand [L] and estimated electric energy demand [kWh].
    active_routes_at, routes_departing_at : dict
        Route identifiers active or departing at each timestep.
    overlapping_routes : list of tuple
        Pairs of routes whose time intervals overlap within a day.
    route_pre_departure : dict
        Timestep used for the battery's pre-departure energy requirement.
    """
    
    def __init__(self, route_csv, duration_days, times):
        """Load the route schedule and construct all derived mappings."""
        self.routes_df = pd.read_csv(route_csv)
        self.duration_days = duration_days
        self.times = times
        self.process_routes()

    # ============================================================
    # Convert HH:MM to decimal hours
    # ============================================================
    def time_to_hours(self, time):
        """Convert an ``HH:MM`` clock-time value to decimal hours.
        
        Parameters
        ----------
        time : str
            Time formatted as hours and minutes, e.g. ``07:30``.
        
        Returns
        -------
        float
            Decimal hours from midnight.
        """
        
        hours, minutes = str(time).split(":")
        return int(hours) + int(minutes) / 60

    # ============================================================
    # Process Route Data
    # ============================================================
    def process_routes(self):

        """Expand the daily schedule and populate optimisation-ready mappings.
        
        Departure times are rounded down and return times up to whole hours.
        Routes are treated as active from departure (inclusive) to return
        (exclusive). Each repeated route receives a unique integer identifier.
        """
        # Round departures down so routes occupy the entire starting hour.
        self.routes_df["departure_hour"] = (self.routes_df["Pick Up Time"].apply(self.time_to_hours).apply(math.floor))
        # Round returns up so partial final hours remain unavailable.
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
        # Map expanded routes to zero-based years (365-day blocks).
        self.route_year = dict(zip(self.routes_df["route_no"].astype(int), self.routes_df["year"].astype(int)))

        # --------------------------------------------------------
        # Routes and Truck Assignment
        # --------------------------------------------------------

        self.routes = (self.routes_df["route_no"].astype(int).tolist())
        self.trucks = sorted(self.routes_df["Truck No"].astype(int).unique().tolist())
        self.route_truck = dict(zip(self.routes_df["route_no"].astype(int), self.routes_df["Truck No"].astype(int)))

        # --------------------------------------------------------
        # Route departure and return times
        # --------------------------------------------------------

        self.route_departure = dict(zip(self.routes_df["route_no"].astype(int), self.routes_df["departure_t"].astype(int)))
        self.route_return = dict(zip(self.routes_df["route_no"].astype(int), self.routes_df["return_t"].astype(int)))

        # --------------------------------------------------------
        # Route energy requirement
        # --------------------------------------------------------

        # Estimated EV battery energy required per litre of diesel replaced.
        # Units: kWh/L diesel
        kwh_per_litre_diesel_equivalent = 4.469729345

        self.route_diesel_used = dict(zip(
            self.routes_df["route_no"].astype(int),
            self.routes_df["Diesel Used"].astype(float)
        ))

        self.route_energy = {
            r: self.route_diesel_used[r] * kwh_per_litre_diesel_equivalent
            for r in self.routes
        }

        # --------------------------------------------------------
        # Routes and Trucks active at each timestep
        # --------------------------------------------------------

        # Build the active-route set at each optimisation timestep.
        self.active_routes_at = {t: [] for t in self.times}

        for r in self.routes:
            for t in range(self.route_departure[r], self.route_return[r]):
                if t in self.active_routes_at:
                    self.active_routes_at[t].append(r)

        self.active_trucks_at = {t: sorted({self.route_truck[r] for r in self.active_routes_at[t]}) for t in self.times}

        # --------------------------------------------------------
        # Overlapping routes
        # --------------------------------------------------------
        # Record route pairs with intersecting time intervals
        # The optimiser can also enforce conflicts through active-route constraints.
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
        # Group departures by timestep for swappable-battery energy deductions.
        self.routes_departing_at = {t: [] for t in self.times}

        for r in self.routes:
            if self.route_departure[r] in self.routes_departing_at:
                self.routes_departing_at[self.route_departure[r]].append(r)

        # --------------------------------------------------------
        # Route pre-departure energy requirement
        # --------------------------------------------------------
        # Use the preceding timestep for pre-departure SOC checks, clamped to the first timestep.
        self.route_pre_departure = {r: max(self.times[0], self.route_departure[r] - 1) for r in self.routes}




