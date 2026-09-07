import pandas as pd
import math


class RouteDataProcessor:

    def __init__(self, route_csv, duration_days, times):

        self.route_csv = route_csv
        self.duration_days = duration_days
        self.times = times

        self.routes_df = pd.read_csv(route_csv)

        self.process_routes()


    def time_to_hours(self, time_value):

        time_value = str(time_value).strip()

        hours, minutes = time_value.split(":")

        return int(hours) + int(minutes) / 60


    def process_routes(self):

        # Rename columns
        self.routes_df = self.routes_df.rename(columns={
            "Route No": "route_no",
            "Pick Up Time": "departure_time",
            "Drop Off Time": "return_time"
        })


        # Convert to decimal hours
        self.routes_df["departure_decimal"] = (
            self.routes_df["departure_time"]
            .apply(self.time_to_hours)
        )

        self.routes_df["return_decimal"] = (
            self.routes_df["return_time"]
            .apply(self.time_to_hours)
        )


        # Round to hourly timestep
        self.routes_df["departure_hour"] = (
            self.routes_df["departure_decimal"]
            .apply(math.floor)
            .astype(int)
        )

        self.routes_df["return_hour"] = (
            self.routes_df["return_decimal"]
            .apply(math.ceil)
            .astype(int)
        )


        # Repeat routes across study duration
        daily_routes = self.routes_df.copy()

        expanded_routes = []

        new_route_no = 1

        for day in range(self.duration_days):

            for _, row in daily_routes.iterrows():

                route = row.copy()

                route["route_no"] = new_route_no
                route["day"] = day

                route["departure_t"] = (
                    int(row["departure_hour"])
                    + day * 24
                )

                route["return_t"] = (
                    int(row["return_hour"])
                    + day * 24
                )

                expanded_routes.append(route)

                new_route_no += 1


        self.routes_df = pd.DataFrame(expanded_routes)


        # Route set
        self.routes = (
            self.routes_df["route_no"]
            .astype(int)
            .tolist()
        )


        # Departure times
        self.route_departure = (
            self.routes_df
            .set_index("route_no")["departure_t"]
            .astype(int)
            .to_dict()
        )


        # Return times
        self.route_return = (
            self.routes_df
            .set_index("route_no")["return_t"]
            .astype(int)
            .to_dict()
        )


        # Active routes at each timestep
        self.active_routes_at = {
            t: []
            for t in self.times
        }

        for r in self.routes:

            for t in range(
                self.route_departure[r],
                self.route_return[r]
            ):

                if t in self.active_routes_at:

                    self.active_routes_at[t].append(r)