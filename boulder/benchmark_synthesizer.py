import datetime
import json
import math
import uuid
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

import numpy as np
from astral import LocationInfo
from astral.sun import sun
from jinja2 import Template

from boulder.domains.db import AttractionDB, HotelDB, RestaurantDB, TrainDB


DB_DIR = Path(__file__).resolve().parent.parent / "data" / "db"

DB_CLASSES = {
    "train_db": TrainDB,
    "hotel_db": HotelDB,
    "restaurant_db": RestaurantDB,
    "attraction_db": AttractionDB,
}


def number_to_word(n: int) -> str:
    words = {
        0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
        6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
        11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
        16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty"
    }
    if n in words:
        return words[n]
    return str(n)


class BenchmarkGenerator(ABC):
    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    @staticmethod
    def _load_db(table: str):
        base = table.split("-")[0]
        cls = DB_CLASSES[base]
        return cls.from_json(str(DB_DIR / f"{table}.json"))

    @abstractmethod
    def generate(self, dialogue_template: list[dict], **kwargs) -> dict | None:
        pass

    def _random_date(self) -> datetime.date:
        return datetime.date(2025, 1, 1) + datetime.timedelta(
            days=int(self.rng.integers(0, 365))
        )

    def _random_time(self) -> datetime.time:
        return datetime.time(
            self.rng.integers(0, 24),
            self.rng.integers(0, 60),
            0,
        )

    def _make_tool_call(self, function_name: str, arguments: dict) -> tuple[dict, str]:
        call_id = str(uuid.uuid4())
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": arguments,
                    },
                }
            ],
        }
        return msg, call_id

    def _make_tool_response(self, call_id: str, content: str) -> dict:
        return {"role": "tool", "id": call_id, "content": content}

    def _render_dialogue(self, dialogue_template: list[dict], placeholders: dict[str, dict], template_vars: dict) -> list[dict]:
        rendered_messages = []
        for message in dialogue_template:
            if message["role"] == "placeholder":
                placeholder_name = message["content"]
                if placeholder_name in placeholders:
                    message_rendered = placeholders[placeholder_name]
                else:
                    raise ValueError(f"Unknown placeholder: {placeholder_name}")
            else:
                content = Template(message["content"]).render(**template_vars)
                message_rendered = {"role": message["role"], "content": content}
            rendered_messages.append(message_rendered)
        return rendered_messages

    def _weighted_regional_sample(self, items_with_distances: list[tuple], n: int = 1, num_regions: int = 5):
        """
        Sample items using weighted regional sampling based on distance.

        Sorts items by distance, divides into regions with exponential decay
        probability (closer = more likely), and samples from selected regions.

        Args:
            items_with_distances: list of (item, distance) tuples
            n: number of items to sample
            num_regions: number of distance-based regions

        Returns:
            If n=1: (item, distance, region_idx)
            If n>1: (items_list, distances_list)
        """
        sorted_items = sorted(items_with_distances, key=lambda x: x[1])

        total_items = len(sorted_items)
        items_per_region = total_items // num_regions
        remainder = total_items % num_regions

        regions = [[] for _ in range(num_regions)]
        idx = 0
        for region_idx in range(num_regions):
            region_size = items_per_region + (1 if region_idx < remainder else 0)
            regions[region_idx] = sorted_items[idx:idx + region_size]
            idx += region_size

        region_probs = np.array([np.exp(-i) for i in range(num_regions)])
        region_probs = region_probs / region_probs.sum()

        if n == 1:
            selected_region_idx = self.rng.choice(num_regions, size=1, p=region_probs)[0]

            if not regions[selected_region_idx]:
                for offset in range(1, num_regions):
                    for direction in [1, -1]:
                        fallback_idx = selected_region_idx + direction * offset
                        if 0 <= fallback_idx < num_regions and regions[fallback_idx]:
                            selected_region_idx = fallback_idx
                            break
                    if regions[selected_region_idx]:
                        break

            chosen_item, chosen_distance = self.rng.choice(
                regions[selected_region_idx], size=1, replace=False
            )[0]
            return chosen_item, chosen_distance, selected_region_idx
        else:
            selected_items = []
            selected_distances = []

            for i in range(n):
                selected_region_idx = self.rng.choice(
                    num_regions, size=1, p=region_probs
                )[0]

                if not regions[selected_region_idx]:
                    for offset in range(1, num_regions):
                        for direction in [1, -1]:
                            fallback_idx = selected_region_idx + direction * offset
                            if 0 <= fallback_idx < num_regions and regions[fallback_idx]:
                                selected_region_idx = fallback_idx
                                break
                        if regions[selected_region_idx]:
                            break

                available_in_region = [
                    (item, dist)
                    for (item, dist) in regions[selected_region_idx]
                    if item not in selected_items
                ]

                if not available_in_region:
                    available_all = [
                        (item, dist)
                        for (item, dist) in sorted_items
                        if item not in selected_items
                    ]
                    if not available_all:
                        raise ValueError("Could not find enough unique items")
                    chosen_item, chosen_distance = self.rng.choice(
                        available_all, size=1, replace=False
                    )[0]
                else:
                    chosen_item, chosen_distance = self.rng.choice(
                        available_in_region, size=1, replace=False
                    )[0]

                selected_items.append(chosen_item)
                selected_distances.append(chosen_distance)

            return selected_items, selected_distances

    @staticmethod
    def _area_adj(area_value: str) -> str:
        return "central" if area_value == "centre" else area_value

    @staticmethod
    def _format_pricerange(pricerange: str) -> str:
        if pricerange == "moderate":
            return "a moderately priced"
        return f"{select_article_for_pricerange(pricerange)} {pricerange}"

    @staticmethod
    def _stars_phrase(stars: str) -> str:
        return f"{stars}-star " if stars not in ["0", "?"] else ""

    def _random_end_time(self, start_hour: int, start_minute: int, end_hour: int) -> tuple[datetime.time, int]:
        if start_hour == end_hour:
            minute_choices = [x for x in [0, 15, 30, 45, 60] if x > start_minute]
        elif end_hour == start_hour + 1:
            minute_choices = [x for x in [0, 15, 30, 45, 60] if x >= start_minute]
        else:
            minute_choices = [0, 15, 30, 45]

        end_minute = self.rng.choice(minute_choices)
        if end_minute == 60:
            end_minute = 0
            end_hour = (end_hour + 1) % 24
        return datetime.time(end_hour, end_minute), end_hour


class TrainPriceGenerator(BenchmarkGenerator):

    def generate(self, dialogue_template: list[dict] | None = None) -> dict:
        train_db = self._load_db("train_db")

        current_date = self._random_date()
        current_time = self._random_time()

        num_people = int(self.rng.integers(2, 7))
        discount_options = [0.20, 0.33, 0.50, 0.60]

        people = []
        for _ in range(num_people):
            ticket_type_value = str(self.rng.choice(["return", "one-way"]))

            if self.rng.random() < 0.5:
                discount = float(self.rng.choice(discount_options))
            else:
                discount = 0.0

            first_class = self.rng.random() < 0.25

            person = {
                "ticket_type": ticket_type_value,
                "discount": discount,
                "first_class": first_class,
            }
            people.append(person)

        train_schema = train_db.get_schema()

        departure = str(self.rng.choice(train_schema["departure"]["enum"], size=1, replace=False)[0])
        destinations = set(train.destination for train in train_db.query(departure=departure))
        destination = str(self.rng.choice(list(destinations), size=1, replace=False)[0])
        weekdays = set(train.day.value for train in train_db.query(departure=departure, destination=destination))
        weekday = str(self.rng.choice(list(weekdays), size=1, replace=False)[0])
        leave_ats = set(train.leave_at for train in train_db.query(departure=departure, destination=destination, weekday=weekday))
        leave_at = str(self.rng.choice(list(leave_ats), size=1, replace=False)[0])

        leave_at_time = datetime.datetime.strptime(leave_at, "%H:%M:%S").time()
        total_minutes = leave_at_time.hour * 60 + leave_at_time.minute
        rounded_minutes = round(total_minutes / 5) * 5
        if total_minutes >= rounded_minutes:
            start_minutes = rounded_minutes
            end_minutes = rounded_minutes + 10
        else:
            start_minutes = rounded_minutes - 10
            end_minutes = rounded_minutes

        start_time = f"{start_minutes // 60:02d}:{start_minutes % 60:02d}"
        end_time = f"{end_minutes // 60:02d}:{end_minutes % 60:02d}"

        train_parameters = {
            "departure": departure,
            "destination": destination,
            "weekday": weekday,
            "leave_before": leave_at[:-3],
            "leave_after": leave_at[:-3],
        }
        train = train_db.query(**train_parameters)[0]
        train_json = json.dumps(train.to_dict())

        person_string = " " + self._generate_person_descriptions(people)

        total_price = 0
        for person in people:
            unit_price = train.get_price_as_float(is_first_class=person["first_class"])
            if person["ticket_type"] == "return":
                total_price += unit_price * 2 * (1 - person["discount"])
            else:
                total_price += unit_price * (1 - person["discount"])

        prompt = f"Given the trains in JSON format below, compute the total price for train tickets for {number_to_word(num_people)} people.{person_string}\nA return ticket costs exactly twice the price of a single ticket.\n\nTrains:\n{train_json}"

        tool_call_trains, tool_call_trains_id = self._make_tool_call("find_train", train_parameters)
        tool_response_trains = self._make_tool_response(tool_call_trains_id, train_json)

        departure_capitalized = capitalize_name(departure)
        destination_capitalized = capitalize_name(destination)
        weekday_capitalized = weekday.capitalize()

        placeholders = {
            "tool_call_trains": tool_call_trains,
            "tool_response_trains": tool_response_trains,
        }
        template_vars = {
            "departure": departure_capitalized,
            "destination": destination_capitalized,
            "weekday": weekday_capitalized,
            "start_time": start_time,
            "end_time": end_time,
            "leave_at": leave_at[:-3],
            "num_people": number_to_word(num_people),
            "person_string": person_string,
        }
        rendered_messages = self._render_dialogue(dialogue_template, placeholders, template_vars)

        return {
            "messages": [{"role": "system", "content": prompt}] + rendered_messages,
            "current_date": current_date.isoformat(),
            "current_time": current_time.isoformat("minutes"),
            "trains_params": train_parameters,
            "people": people,
            "trains": [train.to_dict() for train in train_db.query(**train_parameters)],
            "total_price": total_price,
        }

    @staticmethod
    def _add_discount_if_any(discount: float) -> str:
        if discount > 0:
            return f" with a {int(discount * 100)}% discount"
        return ""

    def _ticket_description(self, person: dict) -> str:
        parts = ["first class"] if person["first_class"] else []
        parts += [person["ticket_type"], "ticket"]
        return f"a {' '.join(parts)}" + self._add_discount_if_any(person["discount"])

    def _describe_person_simple(self, person: dict) -> str:
        verb = self.rng.choice(["am buying", "need", "would like"])
        return f"I {verb} {self._ticket_description(person)}"

    def _describe_single_person(self, person: dict) -> str:
        subject = self.rng.choice(["another person", "one more person"])
        verb = self.rng.choice(["is buying", "needs", "wants"])
        return f"{subject} {verb} {self._ticket_description(person)}"

    def _describe_ticket_type_group(self, members: list[tuple[int, dict]], ticket_type: str, group_nouns: list[str], includes_first: bool, total_people: int) -> str:
        n = len(members)

        if n == total_people and total_people == 2:
            group_start = "both"
        elif n == total_people and total_people > 2:
            group_start = "all of them"
        elif includes_first:
            if n == 2:
                group_start = self.rng.choice(["my colleague and I", "a friend and I", "two of us"])
            else:
                group_start = f"{number_to_word(n)} {self.rng.choice(group_nouns)}"
        else:
            group_start = f"{number_to_word(n)} {self.rng.choice(group_nouns)}"

        verb = self.rng.choice(["are buying", "need", "want"])

        base_desc = f"{group_start} {verb} {ticket_type} tickets"

        first_class_members = [(i, p) for i, p in members if p["first_class"]]
        standard_class_members = [(i, p) for i, p in members if not p["first_class"]]

        all_same_class = len(first_class_members) == 0 or len(standard_class_members) == 0
        all_same_discount = len(set(p["discount"] for i, p in members)) == 1

        if all_same_class and all_same_discount:
            is_first_class = len(first_class_members) > 0
            discount = members[0][1]["discount"]
            quantifier = "both" if n == 2 else "all"

            if is_first_class and discount > 0:
                base_desc += f", {quantifier} of them first class with {int(discount * 100)}% discounts"
            elif is_first_class:
                base_desc += f", {quantifier} of them first class"
            elif discount > 0:
                base_desc += f", {quantifier} with {int(discount * 100)}% discounts"
            return base_desc

        class_descriptions_with_priority = []
        multiple_class_groups = len(first_class_members) > 0 and len(standard_class_members) > 0

        for class_members, class_type in [(first_class_members, "first class"), (standard_class_members, None)]:
            if not class_members:
                continue
            desc = self._describe_class_group(class_members, n, class_type, multiple_class_groups)
            if desc:
                has_breakdown = len(set(p["discount"] for _, p in class_members)) > 1
                class_descriptions_with_priority.append((has_breakdown, desc))

        class_descriptions_with_priority.sort(key=lambda x: x[0])
        class_descriptions = [desc for _, desc in class_descriptions_with_priority]

        if class_descriptions:
            if len(class_descriptions) == 1:
                return base_desc + ", " + class_descriptions[0]
            return base_desc + ", " + ", ".join(class_descriptions[:-1]) + ", and " + class_descriptions[-1]

        return base_desc

    def _describe_class_group(self, class_members: list[tuple[int, dict]], total_n: int, class_type: str, multiple_class_groups: bool) -> str | None:
        n = len(class_members)

        discount_groups: dict[float, list] = defaultdict(list)
        for i, p in class_members:
            discount_groups[p["discount"]].append((i, p))

        if n == 1:
            if class_type:
                return f"one {class_type}" + self._add_discount_if_any(class_members[0][1]["discount"])
            else:
                if multiple_class_groups:
                    ticket_class = self.rng.choice(["standard class", "regular class"])
                    return f"one {ticket_class}" + self._add_discount_if_any(class_members[0][1]["discount"])
                else:
                    if class_members[0][1]["discount"] > 0:
                        return f"one with a {int(class_members[0][1]['discount'] * 100)}% discount"
                    return None
        else:
            if multiple_class_groups:
                if class_type:
                    ref = number_to_word(n)
                else:
                    ticket_type = self.rng.choice(["standard tickets", "regular tickets"])
                    ref = f"{number_to_word(n)} {ticket_type}"
            else:
                if n == total_n:
                    ref = "both of them" if n == 2 else "all of them"
                else:
                    ref = f"{number_to_word(n)} of them"

            if class_type:
                base = f"{ref} {class_type}"
            else:
                base = ref

            if len(discount_groups) == 1:
                discount = list(discount_groups.keys())[0]
                if discount > 0:
                    quantifier = "both" if n == 2 else "all"
                    return f"{base}, {quantifier} with {int(discount * 100)}% discounts"
                return base

            discount_descs = []
            count = 0
            for discount, disc_members in sorted(discount_groups.items(), reverse=True):
                disc_count = len(disc_members)
                if disc_count == 1:
                    if count == 0:
                        ref_inner = "one"
                    elif count == 1 and n == 2:
                        ref_inner = "the other"
                    else:
                        ref_inner = self.rng.choice(["another", "one"])

                    if discount > 0:
                        discount_descs.append(f"{ref_inner} with a {int(discount * 100)}% discount")
                    else:
                        discount_descs.append(f"{ref_inner} without discount")
                else:
                    if discount > 0:
                        discount_descs.append(f"{number_to_word(disc_count)} with {int(discount * 100)}% discounts")
                    else:
                        discount_descs.append(f"{number_to_word(disc_count)} without discounts")
                count += disc_count

            if discount_descs:
                if multiple_class_groups or class_type:
                    return f"{base} - {', '.join(discount_descs)}"
                return ', '.join(discount_descs)
            return base

    def _generate_person_descriptions(self, people: list[dict]) -> str:
        num_people = len(people)

        group_nouns = ["colleagues", "of us", "people", "travelers"]

        ticket_type_groups: dict[str, list] = defaultdict(list)
        for i, person in enumerate(people):
            ticket_type_groups[person["ticket_type"]].append((i, person))

        sentences = []
        processed = set()

        first_person_ticket_type = people[0]["ticket_type"]
        first_person_group = ticket_type_groups[first_person_ticket_type]

        if len(first_person_group) == 1:
            first_desc = self._describe_person_simple(people[0])
            sentences.append(first_desc)
            processed.add(0)

        ticket_types = ["return", "one-way"]
        if first_person_ticket_type == "one-way":
            ticket_types = ["one-way", "return"]

        for ticket_type in ticket_types:
            if ticket_type not in ticket_type_groups:
                continue

            group_members = ticket_type_groups[ticket_type]
            unprocessed = [(i, p) for i, p in group_members if i not in processed]

            if len(unprocessed) == 0:
                continue
            elif len(unprocessed) == 1:
                i, person = unprocessed[0]
                desc = self._describe_single_person(person)
                sentences.append(desc)
                processed.add(i)
            else:
                includes_first = 0 in [i for i, _ in unprocessed]
                desc = self._describe_ticket_type_group(unprocessed, ticket_type, group_nouns, includes_first, num_people)
                sentences.append(desc)
                processed.update([i for i, _ in unprocessed])

        formatted_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if not sent.endswith('.'):
                sent += '.'
            if sent and sent[0].islower():
                sent = sent[0].upper() + sent[1:]
            formatted_sentences.append(sent)

        return " ".join(formatted_sentences)


class TrainSunsetGenerator(BenchmarkGenerator):

    CITY_COORDINATES = {
        "peterborough": (52.5726, -0.2427),
        "london liverpool street": (51.5182, -0.0814),
        "ely": (52.3990, 0.2620),
        "cambridge": (52.2055, 0.1187),
        "kings lynn": (52.7539, 0.3955),
        "bishops stortford": (51.8773, 0.1511),
        "stansted airport": (51.8900, 0.2615),
        "london kings cross": (51.5074, -0.1278),
        "stevenage": (51.9017, -0.2027),
        "leicester": (52.6362, -1.1332),
        "broxbourne": (51.7466, -0.0191),
        "norwich": (52.6286, 1.2924),
        "birmingham new street": (52.4776, -1.8987),
    }

    def generate(self, dialogue_template: list[dict] | None = None, include_sunset: bool = False) -> dict | None:
        train_db = self._load_db("train_db")

        departure_hours_before_sunset = self.rng.integers(1, 5)
        minutes_precision_query = 3

        current_date = self._random_date()
        current_time = self._random_time()
        offset_hours = int(departure_hours_before_sunset + 1)

        train_schema = train_db.get_schema()
        departure_value = str(self.rng.choice(train_schema["departure"]["enum"], size=1, replace=False)[0])
        destination_stations = {train.destination for train in train_db.query(departure=departure_value, weekday=current_date.strftime("%A").lower())}
        destination_value = self.rng.choice(list(destination_stations), size=1, replace=False)[0]
        trains = train_db.query(departure=departure_value, destination=destination_value, weekday=current_date.strftime("%A").lower())

        # Compute sunset time for the date and location
        latitude, longitude = self.CITY_COORDINATES[destination_value]
        city = LocationInfo("Cambridge", "England", "Europe/London", latitude, longitude)
        sunset = sun(city.observer, date=current_date, tzinfo=city.timezone)
        sunset_time = sunset["sunset"]

        # Randomly select earliest departure time
        departure_offset_minutes = self._random_minutes(minutes_precision_query)
        earliest_departure_time = sunset_time - datetime.timedelta(hours=offset_hours, minutes=departure_offset_minutes)

        # Randomly select current time
        current_time_min_hour = 5
        current_time_offset_minutes = self._random_minutes(minutes_precision_query)
        current_time_max_hour = (earliest_departure_time - datetime.timedelta(hours=1, minutes=current_time_offset_minutes)).hour
        current_time_hour = self.rng.integers(current_time_min_hour, current_time_max_hour)
        current_time_minute = self.rng.integers(0, 60)
        current_time = datetime.time(current_time_hour, current_time_minute, 0)

        sunset_time = sunset_time.time()
        earliest_departure_time = earliest_departure_time.time()
        earliest_departure_time = datetime.time(
            earliest_departure_time.hour,
            earliest_departure_time.minute // 5 * 5,
            0
        )

        all_trains = train_db.query(
            destination=destination_value,
            departure=departure_value,
            leave_after=earliest_departure_time.isoformat("minutes"),
            weekday=current_date.strftime("%A").lower(),
        )

        all_trains = [
            train
            for train in all_trains
            if train.leave_at >= earliest_departure_time
        ]

        matching_trains = [
            train
            for train in all_trains
            if train.arrive_by <= sunset_time
            and train.leave_at.hour < train.arrive_by.hour
        ]
        if not matching_trains:
            return None

        last_departure_time = max(train.leave_at for train in matching_trains)

        all_trains_list = [train.to_dict() for train in all_trains]
        matching_trains_list = [train.to_dict() for train in matching_trains]

        prompt = f"Given the list of trains in JSON format below, select departure time of the latest train that arrives in {capitalize_name(destination_value)} before sunset.\n\nCurrent date is {current_date.isoformat()} and current time is {current_time.isoformat('minutes')}\n\nTrains:\n{json.dumps(all_trains_list)}"
        if include_sunset:
            prompt += f" Sunset time is {sunset_time.isoformat('minutes')}."

        num_trains = len(all_trains_list)
        found_trains_phrase = f"I found {number_to_word(num_trains)} trains" if num_trains > 1 else "I found one train"

        tool_call_trains, tool_call_trains_id = self._make_tool_call(
            "search_trains", {"departure": departure_value, "destination": destination_value, "weekday": current_date.strftime("%A").lower()})
        tool_response_trains = self._make_tool_response(tool_call_trains_id, json.dumps(all_trains_list))

        departure_capitalized = capitalize_name(departure_value)
        destination_capitalized = capitalize_name(destination_value)

        placeholders = {
            "tool_call_trains": tool_call_trains,
            "tool_response_trains": tool_response_trains,
        }
        template_vars = {
            "departure": departure_capitalized,
            "destination": destination_capitalized,
            "earliest_departure_time": earliest_departure_time.isoformat("minutes"),
            "found_trains_phrase": found_trains_phrase,
        }
        rendered_messages = self._render_dialogue(dialogue_template, placeholders, template_vars)

        return {
            "messages": [{"role": "system", "content": prompt}] + rendered_messages,
            "current_date": current_date.isoformat(),
            "current_time": current_time.isoformat("minutes"),
            "earliest_departure_time": earliest_departure_time.isoformat("minutes"),
            "sunset_time": sunset_time.isoformat("minutes"),
            "departure_value": departure_value,
            "matching_trains": matching_trains_list,
            "last_departure_time": last_departure_time.isoformat("minutes"),
            "all_trains": all_trains_list,
        }

    def _random_minutes(self, precision: int) -> int:
        if precision == 1:
            return 0
        elif precision == 2:
            return int(self.rng.choice([0, 15, 30, 45]))
        elif precision == 3:
            return int(self.rng.choice(range(0, 60, 5)))
        elif precision == 4:
            return int(self.rng.integers(0, 60))
        else:
            raise ValueError("Invalid difficulty level for minutes precision")


class RestaurantOpenTimeGenerator(BenchmarkGenerator):

    INTERVAL_RELATIONS = [
        "query_equals_open_hours",
        "query_starts_open_hours",
        "query_finishes_open_hours",
        "query_precedes_open_hours",
        "query_is_preceded_by_open_hours",
        "query_meets_open_hours",
        "query_is_met_by_open_hours",
        "query_contains_open_hours",
        "query_during_open_hours",
        "query_overlaps_with_hours",
        "query_is_overlapped_by_hours",
        "query_is_started_by_hours",
        "query_is_finished_by_hours",
    ]

    RELATION_NAMES = {
        "query_equals_open_hours": "equals",
        "query_starts_open_hours": "starts",
        "query_finishes_open_hours": "finishes",
        "query_precedes_open_hours": "precedes",
        "query_is_preceded_by_open_hours": "is_preceded_by",
        "query_meets_open_hours": "meets",
        "query_is_met_by_open_hours": "is_met_by",
        "query_contains_open_hours": "contains",
        "query_during_open_hours": "during",
        "query_overlaps_with_hours": "overlaps",
        "query_is_overlapped_by_hours": "is_overlapped_by",
        "query_is_started_by_hours": "is_started_by",
        "query_is_finished_by_hours": "is_finished_by",
    }

    def generate(self, dialogue_template: list[dict] | None = None) -> dict:
        restaurant_db = self._load_db("restaurant_db")

        current_date = self._random_date()
        current_time = self._random_time()

        num_restaurants = 0
        while num_restaurants < 2:
            restaurant = self.rng.choice(restaurant_db.query(), size=1, replace=False)[0]
            food_value = restaurant.food
            area_value = restaurant.area
            restaurants_area = restaurant_db.query(food=food_value, area=area_value)
            num_restaurants = len(restaurants_area)

        # Randomly select interval relation
        min_start_hour = 5
        max_start_hour = 21
        interval_relation = self.rng.choice(self.INTERVAL_RELATIONS, size=1, replace=False)[0]
        restaurant_with_matching_hours = self.rng.choice(restaurants_area, size=1, replace=False)[0]

        # Randomly select weekday
        weekday = self.rng.choice(list(restaurant_with_matching_hours.openhours.keys()), size=1, replace=False)[0]

        opening_time = datetime.time.fromisoformat(restaurant_with_matching_hours.openhours[weekday]["open"])
        closing_time = datetime.time.fromisoformat(restaurant_with_matching_hours.openhours[weekday]["close"])

        query = self._build_query(
            interval_relation,
            opening_time, closing_time,
            min_start_hour, max_start_hour,
        )

        matching_restaurants_list = [
            restaurant.to_dict()
            for restaurant in restaurants_area
            if self._open_hours_match(query, restaurant.openhours, weekday)
        ]
        matching_restaurants_overlap_list = [
            restaurant.to_dict()
            for restaurant in restaurants_area
            if self._open_hours_match(query, restaurant.openhours, weekday, strict=False)
        ]
        all_restaurants_list = [restaurant.to_dict() for restaurant in restaurants_area]

        restaurant_params = {
            "food": food_value,
            "area": area_value.value,
        }
        query["hours"] = {key: hour.isoformat("minutes") for key, hour in query["hours"].items()}

        query_start, query_end = query["hours"]["start"], query["hours"]["end"]
        strict_overlap = len(matching_restaurants_list) > 0
        strict_overlap_specification = "for the entire time " if strict_overlap else "for at least an hour "
        time_specification = f"{strict_overlap_specification}between {query_start} and {query_end}"

        prompt = f"Given the list of restaurants in JSON format below, select those that are open on {weekday.capitalize()} {time_specification}:\n\nRestaurants:\n{all_restaurants_list}"

        tool_call_restaurant, tool_call_restaurant_id = self._make_tool_call(
            "search_restaurants", {"food": food_value, "area": area_value.value})
        tool_response_restaurant = self._make_tool_response(tool_call_restaurant_id, json.dumps(all_restaurants_list))

        area_adj = self._area_adj(area_value.value)

        placeholders = {
            "tool_call_restaurant": tool_call_restaurant,
            "tool_response_restaurant": tool_response_restaurant,
        }
        template_vars = {
            "food": food_value,
            "num_restaurants": number_to_word(len(all_restaurants_list)),
            "area": area_value.value,
            "area_adj": area_adj,
            "weekday": weekday.capitalize(),
            "time_specification": time_specification,
        }
        rendered_messages = self._render_dialogue(dialogue_template, placeholders, template_vars)

        return {
            "messages": [{"role": "system", "content": prompt}] + rendered_messages,
            "current_date": current_date.isoformat(),
            "current_time": current_time.isoformat("minutes"),
            "restaurant_params": restaurant_params,
            "query": query,
            "weekday": weekday,
            "strict_overlap": strict_overlap,
            "reference_restaurant": restaurant_with_matching_hours.to_dict(),
            "matching_restaurants": matching_restaurants_list if strict_overlap else matching_restaurants_overlap_list,
            "all_restaurants": all_restaurants_list,
        }

    @staticmethod
    def _hours_between(t1: datetime.time, t2: datetime.time) -> float:
        return (
            datetime.datetime.combine(datetime.date.min, t2) -
            datetime.datetime.combine(datetime.date.min, t1)
        ).total_seconds() / 3600

    @staticmethod
    def _hours_to_midnight(t: datetime.time) -> float:
        return 24.0 - RestaurantOpenTimeGenerator._hours_between(datetime.time(0, 0), t)

    def _open_hours_match(self, query: dict, openhours: dict, weekday: str, strict: bool = True) -> bool:
        if weekday not in openhours:
            return False
        query_hours = query["hours"]
        restaurant_hours = {k: datetime.time.fromisoformat(v) for k, v in openhours[weekday].items()}

        query_start = query_hours["start"]
        query_end = query_hours["end"]
        rest_open = restaurant_hours["open"]
        rest_close = restaurant_hours["close"]

        query_crosses = query_start > query_end
        rest_crosses = rest_open > rest_close

        # Case 1: Neither crosses midnight
        if not query_crosses and not rest_crosses:
            if strict:
                return query_start >= rest_open and query_end <= rest_close
            overlap_start = max(query_start, rest_open)
            overlap_end = min(query_end, rest_close)
            if overlap_start >= overlap_end:
                return False
            return self._hours_between(overlap_start, overlap_end) >= 1.0

        # Case 2: Query crosses midnight, restaurant doesn't
        elif query_crosses and not rest_crosses:
            if strict:
                return False
            total = 0.0
            # Evening overlap
            if rest_close > query_start:
                evening_start = max(rest_open, query_start)
                if evening_start < rest_close:
                    total += self._hours_between(evening_start, rest_close)
            # Morning overlap
            if rest_open < query_end:
                morning_end = min(rest_close, query_end)
                if rest_open < morning_end:
                    total += self._hours_between(rest_open, morning_end)
            return total >= 1.0

        # Case 3: Restaurant crosses midnight, query doesn't
        elif rest_crosses and not query_crosses:
            if strict:
                in_evening = query_start >= rest_open and query_end >= rest_open
                in_morning = query_start <= rest_close and query_end <= rest_close
                return in_evening or in_morning
            total = 0.0
            # Evening overlap with [rest_open, 24:00)
            if query_end > rest_open:
                evening_start = max(query_start, rest_open)
                if evening_start < query_end:
                    total += self._hours_between(evening_start, query_end)
            # Morning overlap with [00:00, rest_close]
            if query_start < rest_close:
                morning_end = min(query_end, rest_close)
                if query_start < morning_end:
                    total += self._hours_between(query_start, morning_end)
            return total >= 1.0

        # Case 4: Both cross midnight
        else:
            if strict:
                return rest_open <= query_start and rest_close >= query_end
            evening_start = max(rest_open, query_start)
            morning_end = min(rest_close, query_end)
            total = self._hours_to_midnight(evening_start) + self._hours_between(datetime.time(0, 0), morning_end)
            return total >= 1.0

    def _rand_hour(self, low: int, high: int) -> int:
        return int(self.rng.integers(low, high)) % 24

    def _rand_quarter_time(self, hour: int) -> datetime.time:
        return datetime.time(hour, self.rng.choice([0, 15, 30, 45]))

    def _rand_minute_at_least_1h_before(self, hour: int, ref: datetime.time) -> int:
        if hour == ref.hour - 1:
            return self.rng.choice([x for x in [0, 15, 30, 45] if x <= ref.minute])
        return self.rng.choice([0, 15, 30, 45])

    def _rand_minute_at_least_1h_after(self, hour: int, ref: datetime.time) -> int:
        if hour == ref.hour + 1 or (ref.hour >= 23 and hour == (ref.hour + 1) % 24):
            return self.rng.choice([x for x in [0, 15, 30, 45] if x >= ref.minute])
        return self.rng.choice([0, 15, 30, 45])

    def _build_query(
        self,
        interval_relation: str,
        opening_time: datetime.time,
        closing_time: datetime.time,
        min_start_hour: int,
        max_start_hour: int,
    ) -> dict:
        relation = self.RELATION_NAMES.get(interval_relation)
        if relation is None:
            raise ValueError(f"Invalid interval relation: {interval_relation}")

        o, c = opening_time, closing_time

        if relation == "equals":
            start, end = o, c

        elif relation == "starts":
            eh = self._rand_hour(o.hour + 1, c.hour - 1)
            end = datetime.time(eh, self._rand_minute_at_least_1h_after(eh, o))
            start = o

        elif relation == "finishes":
            sh = self._rand_hour(o.hour + 1, c.hour - 1)
            start = datetime.time(sh, self._rand_minute_at_least_1h_before(sh, c))
            end = c

        elif relation == "precedes":
            eh = self._rand_hour(max(min_start_hour + 2, o.hour - 3), o.hour - 1)
            em = self.rng.choice([0, 15, 30, 45])
            end = datetime.time(eh, em)
            sh = self._rand_hour(min_start_hour, eh - 1)
            if sh == eh:
                choices = [x for x in [0, 15, 30, 45, 60] if x < em]
            elif eh == sh + 1:
                choices = [x for x in [0, 15, 30, 45] if x <= em]
            else:
                choices = [0, 15, 30, 45]
            sm = self.rng.choice(choices)
            start = datetime.time(sh, sm)

        elif relation == "is_preceded_by":
            sh = self._rand_hour(c.hour + 1, max(c.hour + 3, max_start_hour))
            sm = self.rng.choice([0, 15, 30, 45])
            start = datetime.time(sh, sm)
            eh = self._rand_hour(sh + 1, sh + 3)
            end, _ = self._random_end_time(sh, sm, eh)

        elif relation == "meets":
            sh = self._rand_hour(min_start_hour, o.hour - 1)
            start = datetime.time(sh, self._rand_minute_at_least_1h_before(sh, o))
            end = o

        elif relation == "is_met_by":
            eh = self._rand_hour(c.hour + 1, c.hour + 3)
            end = datetime.time(eh, self._rand_minute_at_least_1h_after(eh, c))
            start = c

        elif relation == "contains":
            sh = self._rand_hour(min_start_hour, o.hour - 1)
            sm = self.rng.choice([0, 15, 30, 45])
            start = datetime.time(sh, sm)
            eh = self._rand_hour(c.hour + 1, c.hour + 3)
            end, _ = self._random_end_time(sh, sm, eh)

        elif relation == "during":
            sh = self._rand_hour(o.hour, c.hour - 2)
            sm = self.rng.choice([0, 15, 30, 45])
            start = datetime.time(sh, sm)
            eh = self._rand_hour(sh + 1, c.hour - 1)
            end, _ = self._random_end_time(sh, sm, eh)

        elif relation == "overlaps":
            sh = self._rand_hour(min_start_hour, o.hour - 1)
            sm = self.rng.choice([0, 15, 30, 45])
            start = datetime.time(sh, sm)
            eh = self._rand_hour(o.hour + 1, c.hour - 1)
            end, _ = self._random_end_time(sh, sm, eh)

        elif relation == "is_overlapped_by":
            sh = self._rand_hour(o.hour + 1, c.hour - 1)
            sm = self.rng.choice([0, 15, 30, 45])
            start = datetime.time(sh, sm)
            eh = self._rand_hour(c.hour + 1, c.hour + 3)
            end, _ = self._random_end_time(sh, sm, eh)

        elif relation == "is_started_by":
            eh = self._rand_hour(c.hour + 1, c.hour + 3)
            end = self._rand_quarter_time(eh)
            start = o

        elif relation == "is_finished_by":
            sh = self._rand_hour(min_start_hour, o.hour - 1)
            start = self._rand_quarter_time(sh)
            end = c

        return {
            "operator": "between",
            "relation": relation,
            "hours": {"start": start, "end": end},
        }


class TrainFrequencyGenerator(BenchmarkGenerator):

    def generate(self, dialogue_template: list[dict] | None = None) -> dict | None:
        train_db = self._load_db("train_db-extended")

        current_date = self._random_date()
        current_time = self._random_time()

        train = self.rng.choice(train_db.query(), size=1, replace=False)[0]
        weekday = train.day.value
        departure = train.departure
        destination = train.destination

        # Get all trains for this route on this weekday
        trains = train_db.query(departure=departure, destination=destination, weekday=weekday)
        if len(trains) < 2:
            return None

        # Randomly select a query interval
        day_periods = trains[0].metadata["periods"]
        selected_period = self.rng.choice(day_periods, size=1, replace=False)[0]
        query_interval_start = datetime.time.fromisoformat(selected_period["start"])
        query_interval_end = datetime.time.fromisoformat(selected_period["end"])
        average_interval = selected_period["avg_interval_minutes"]
        departure_times = [
            train.leave_at.isoformat("minutes")
            for train in trains if train.leave_at >= query_interval_start and train.leave_at <= query_interval_end
        ]
        trains_sorted = sorted(trains, key=lambda x: x.leave_at)

        trains_list = [train.to_dict() for train in trains_sorted]
        train_json = json.dumps(trains_list)

        start_time = query_interval_start.isoformat("minutes")
        end_time = query_interval_end.isoformat("minutes")
        if end_time == "23:59":
            end_time = "midnight"

        prompt = f"Given the list of trains in JSON format below, calculate how often on average do trains run between {start_time} and {end_time}.\n\nTrains:\n{train_json}"

        num_trains = len(trains_list)
        found_trains_phrase = f"I found {num_trains} trains" if num_trains > 1 else "I found 1 train"

        tool_call_trains, tool_call_trains_id = self._make_tool_call(
            "search_trains", {"departure": departure, "destination": destination, "weekday": weekday})
        tool_response_trains = self._make_tool_response(tool_call_trains_id, train_json)

        departure_capitalized = capitalize_name(departure)
        destination_capitalized = capitalize_name(destination)
        weekday_capitalized = weekday.capitalize()

        placeholders = {
            "tool_call_trains": tool_call_trains,
            "tool_response_trains": tool_response_trains,
        }
        template_vars = {
            "departure": departure_capitalized,
            "destination": destination_capitalized,
            "weekday": weekday_capitalized,
            "found_trains_phrase": found_trains_phrase,
            "query_interval_start": start_time,
            "query_interval_end": end_time,
        }
        rendered_messages = self._render_dialogue(dialogue_template, placeholders, template_vars)

        return {
            "messages": [{"role": "system", "content": prompt}] + rendered_messages,
            "current_date": current_date.isoformat(),
            "current_time": current_time.isoformat("minutes"),
            "weekday": weekday,
            "departure": departure,
            "destination": destination,
            "trains": trains_list,
            "targets": {
                "departure_times": departure_times,
                "average_interval_minutes": average_interval,
            },
        }


class DirectionalRelationsGenerator(BenchmarkGenerator):

    def generate(self, dialogue_template: list[dict] | None = None) -> dict:
        restaurant_db = self._load_db("restaurant_db")
        attraction_db = self._load_db("attraction_db")

        # Select a random restaurant
        restaurants_all = restaurant_db.query()
        restaurant = self.rng.choice(restaurants_all, size=1, replace=False)[0]

        restaurant_area = restaurant.area.value
        restaurant_food = restaurant.food
        restaurant_pricerange = restaurant.pricerange.value
        restaurants_set = restaurant_db.query(food=restaurant_food, area=restaurant_area)

        # Select a random attraction
        attractions_all = attraction_db.query()

        # Filter attractions with valid locations
        attractions_with_location = [a for a in attractions_all if a.area == restaurant.area and a.location is not None and restaurant.location is not None]
        if not attractions_with_location:
            raise ValueError("No attractions with valid locations found")

        attraction = self.rng.choice(attractions_with_location, size=1, replace=False)[0]

        # Select a random direction to ask about
        possible_directions = ['north', 'south', 'east', 'west']
        asked_direction = self.rng.choice(possible_directions, size=1, replace=False)[0]

        # Determine if the asked direction is correct
        is_correct = is_direction_correct(asked_direction, restaurant.location, attraction.location)

        restaurants_set_json = json.dumps([r.to_dict() for r in restaurants_set])
        attractions_set_json = json.dumps([attraction.to_dict()])

        units_info = " Spatial coordinates are given in meters from the origin, a (0, 0) point in the southwest corner of the map."

        prompt = f"Given the restaurants and attractions in JSON format below, determine if {capitalize_name(attraction.name)} is {asked_direction} of {capitalize_name(restaurant.name)}.{units_info}\n\nRestaurants:\n{restaurants_set_json}\n\nAttractions:\n{attractions_set_json}"

        current_date = self._random_date()
        current_time = self._random_time()

        tomorrow_date = current_date + datetime.timedelta(days=1)
        tomorrow_weekday = tomorrow_date.strftime('%A').lower()

        if tomorrow_weekday in restaurant.openhours:
            opening_time = datetime.time.fromisoformat(restaurant.openhours[tomorrow_weekday]["open"])
            closing_time = datetime.time.fromisoformat(restaurant.openhours[tomorrow_weekday]["close"])
            latest_hour = max(opening_time.hour, closing_time.hour - 1)
            booking_hour = int(self.rng.integers(opening_time.hour, latest_hour + 1))
            booking_minute = self.rng.choice([0, 15, 30, 45])
            booking_time = datetime.time(booking_hour, booking_minute)
        else:
            booking_time = datetime.time(19, 0)

        tool_call_restaurant, tool_call_restaurant_id = self._make_tool_call(
            "search_restaurants", {"food": restaurant_food, "area": restaurant_area})
        tool_response_restaurant = self._make_tool_response(tool_call_restaurant_id, restaurants_set_json)

        tool_call_booking, tool_call_booking_id = self._make_tool_call(
            "book_restaurant", {
                "restaurant_id": restaurant.id,
                "date": tomorrow_date.isoformat(),
                "time": booking_time.isoformat("minutes"),
                "num_people": 2,
            })
        tool_response_booking = self._make_tool_response(
            tool_call_booking_id, json.dumps({"success": True, "message": "Restaurant booked successfully"}))

        tool_call_attraction, tool_call_attraction_id = self._make_tool_call(
            "search_attractions", {"name": attraction.name})
        tool_response_attraction = self._make_tool_response(tool_call_attraction_id, attractions_set_json)

        restaurant_pricerange_word = "moderately priced" if restaurant_pricerange == "moderate" else restaurant_pricerange
        restaurant_area_adj = self._area_adj(restaurant_area)
        attraction_area_adj = self._area_adj(attraction.area.value)
        if attraction.entrance_fee == "free":
            attraction_fee_info = " The entrance is free of charge."
        elif attraction.entrance_fee == "?":
            attraction_fee_info = ""
        else:
            attraction_fee_info = f" The entrance fee is {attraction.entrance_fee}."

        placeholders = {
            "tool_call_restaurant": tool_call_restaurant,
            "tool_response_restaurant": tool_response_restaurant,
            "tool_call_booking": tool_call_booking,
            "tool_response_booking": tool_response_booking,
            "tool_call_attraction": tool_call_attraction,
            "tool_response_attraction": tool_response_attraction,
        }
        template_vars = {
            "restaurant_name": capitalize_name(restaurant.name),
            "restaurant_pricerange": restaurant_pricerange_word,
            "restaurant_food": capitalize_name(restaurant_food),
            "restaurant_area_adj": restaurant_area_adj,
            "restaurant_address": capitalize_name(restaurant.address),
            "num_restaurants": number_to_word(len(restaurants_set)),
            "booking_time": booking_time.isoformat('minutes'),
            "attraction_name": capitalize_name(attraction.name),
            "attraction_type": attraction.type,
            "attraction_area_adj": attraction_area_adj,
            "attraction_address": capitalize_name(attraction.address),
            "asked_direction": asked_direction,
            "attraction_fee_info": attraction_fee_info,
        }
        rendered_messages = self._render_dialogue(dialogue_template, placeholders, template_vars)

        return {
            "messages": [{"role": "system", "content": prompt}] + rendered_messages,
            "current_date": current_date.isoformat(),
            "current_time": current_time.isoformat("minutes"),
            "restaurant": restaurant.to_dict(),
            "restaurant_params": {
                "area": restaurant_area,
                "food": restaurant_food,
            },
            "attraction": attraction.to_dict(),
            "booking": {
                "date": tomorrow_date.isoformat(),
                "time": booking_time.isoformat("minutes"),
                "num_people": 2
            },
            "targets": {
                "asked_direction": asked_direction,
                "is_correct": is_correct,
            },
        }


class HotelRestaurantDistanceGenerator(BenchmarkGenerator):

    def generate(self, dialogue_template: list[dict] | None = None) -> dict:
        hotel_db = self._load_db("hotel_db")
        restaurant_db = self._load_db("restaurant_db")

        # Select a random hotel
        hotels_all = hotel_db.query()
        hotel = self.rng.choice(hotels_all, size=1, replace=False)[0]

        hotel_area = hotel.area.value
        hotel_pricerange = hotel.pricerange.value
        hotels_set = hotel_db.query(area=hotel_area, pricerange=hotel_pricerange)

        # Select random restaurant using weighted sampling based on distance from hotel
        restaurants_all = restaurant_db.query()

        # Calculate distances from hotel for all restaurants
        restaurant_distances = []
        for restaurant in restaurants_all:
            if restaurant.location is None or hotel.location is None:
                continue
            dist = distance_meters(restaurant.location, hotel.location)
            restaurant_distances.append((restaurant, dist))

        if not restaurant_distances:
            raise ValueError("No restaurants with valid locations found")

        restaurant_chosen, chosen_distance, _ = self._weighted_regional_sample(restaurant_distances)

        restaurant_area = restaurant_chosen.area.value
        restaurant_food = restaurant_chosen.food
        restaurants_set = restaurant_db.query(food=restaurant_food, area=restaurant_area)

        restaurants_set_json = json.dumps([r.to_dict() for r in restaurants_set])
        hotels_set_json = json.dumps([h.to_dict() for h in hotels_set])

        units_info = " Spatial coordinates are given in meters from the origin, a (0, 0) point in the southwest corner of the map."

        prompt = f"Given the hotels and restaurants in JSON format below, calculate the distance from {capitalize_name(restaurant_chosen.name)} to {capitalize_name(hotel.name)}.{units_info}\n\nHotels:\n{hotels_set_json}\n\nRestaurants:\n{restaurants_set_json}"

        tool_call_hotel, tool_call_hotel_id = self._make_tool_call(
            "search_hotels", {"pricerange": hotel_pricerange, "area": hotel_area})
        tool_response_hotel = self._make_tool_response(tool_call_hotel_id, hotels_set_json)

        current_date = self._random_date()
        current_time = self._random_time()
        num_nights = int(self.rng.integers(2, 7))
        checkin_date = current_date + datetime.timedelta(days=1)
        checkout_date = current_date + datetime.timedelta(days=num_nights + 1)

        tool_call_booking, tool_call_booking_id = self._make_tool_call(
            "book_hotel", {
                "hotel_id": hotel.id,
                "rooms": [{
                    "checkin_date": checkin_date.isoformat(),
                    "checkout_date": checkout_date.isoformat(),
                    "room_type": "double",
                    "num_guests": 2,
                }],
            })
        tool_response_booking = self._make_tool_response(
            tool_call_booking_id, json.dumps({"success": True, "message": "Hotel booked successfully"}))

        tool_call_restaurant, tool_call_restaurant_id = self._make_tool_call(
            "search_restaurants", {"food": restaurant_food, "area": restaurant_area})
        tool_response_restaurant = self._make_tool_response(tool_call_restaurant_id, restaurants_set_json)

        hotel_pricerange_word = self._format_pricerange(hotel_pricerange)
        restaurant_pricerange_word = self._format_pricerange(restaurant_chosen.pricerange.value)
        hotel_area_adj = self._area_adj(hotel_area)
        restaurant_area_adj = self._area_adj(restaurant_area)

        stars_phrase = self._stars_phrase(hotel.stars)

        placeholders = {
            "tool_call_hotel": tool_call_hotel,
            "tool_response_hotel": tool_response_hotel,
            "tool_call_booking": tool_call_booking,
            "tool_response_booking": tool_response_booking,
            "tool_call_restaurant": tool_call_restaurant,
            "tool_response_restaurant": tool_response_restaurant,
        }
        template_vars = {
            "hotel_pricerange": hotel_pricerange_word,
            "hotel_area_adj": hotel_area_adj,
            "hotel_name": capitalize_name(hotel.name),
            "stars_phrase": stars_phrase,
            "hotel_type": hotel.type,
            "hotel_address": capitalize_name(hotel.address),
            "num_nights": number_to_word(num_nights),
            "restaurant_food": capitalize_name(restaurant_food),
            "restaurant_area_adj": restaurant_area_adj,
            "restaurant_name": capitalize_name(restaurant_chosen.name),
            "restaurant_pricerange": restaurant_pricerange_word,
            "restaurant_address": capitalize_name(restaurant_chosen.address),
        }
        rendered_messages = self._render_dialogue(dialogue_template, placeholders, template_vars)

        return {
            "messages": [{"role": "system", "content": prompt}] + rendered_messages,
            "current_date": current_date.isoformat(),
            "current_time": current_time.isoformat("minutes"),
            "hotel": hotel.to_dict(),
            "hotel_params": {
                "area": hotel_area,
                "pricerange": hotel_pricerange,
            },
            "restaurant": restaurant_chosen.to_dict(),
            "restaurant_params": {
                "area": restaurant_area,
                "food": restaurant_food,
            },
            "restaurants_all": [restaurant.to_dict() for restaurant in restaurants_set],
            "targets": {
                "distance": chosen_distance,
            },
        }


class HotelAttractionWalkingOrderGenerator(BenchmarkGenerator):

    USER_REQUEST_TEMPLATES_FIRST = [
        "Yes, can you tell me about {attr_name}?",
        "Yes, I'd like to know more about {attr_name}.",
        "Yes, what can you tell me about {attr_name}?",
        "Yes, I'm interested in {attr_name}.",
        "Yes, could you give me some information on {attr_name}?",
        "Yes, I want to learn about {attr_name}.",
        "Yes, can you provide details about {attr_name}?",
        "Yes, tell me about {attr_name}.",
        "Yes, I'd like information on {attr_name}.",
        "Yes, what do you know about {attr_name}?",
    ]

    USER_REQUEST_TEMPLATES_ADDITIONAL = [
        "I'm also interested in {attr_name}.",
        "I'd also like to know about {attr_name}.",
        "Can you also tell me about {attr_name}?",
        "What about {attr_name}?",
        "I also want to learn about {attr_name}.",
        "And {attr_name}?",
        "Could you also give me information on {attr_name}?",
        "I'd also like details on {attr_name}.",
        "Tell me about {attr_name} as well.",
        "I'm also curious about {attr_name}.",
    ]

    def generate(self, dialogue_template: list[dict] | None = None) -> dict:
        hotel_db = self._load_db("hotel_db")
        attraction_db = self._load_db("attraction_db")

        hotels_all = hotel_db.query()
        hotel = self.rng.choice(hotels_all, size=1, replace=False)[0]
        hotel_area = hotel.area.value
        hotel_pricerange = hotel.pricerange.value
        hotels_set = hotel_db.query(area=hotel_area, pricerange=hotel_pricerange)

        attractions_all = attraction_db.query()
        attractions_with_location = [a for a in attractions_all if a.location is not None]

        if len(attractions_with_location) < 2:
            raise ValueError("Not enough attractions with valid locations")

        if hotel.location is None:
            raise ValueError("Hotel has no location")

        # Calculate distances from hotel for all attractions
        attraction_distances = []
        for attraction in attractions_with_location:
            if attraction.location is None:
                continue
            dist = distance_meters(attraction.location, hotel.location)
            attraction_distances.append((attraction, dist))

        if len(attraction_distances) < 2:
            raise ValueError("Not enough attractions with valid locations found")

        # Randomly choose how many attractions to sample (between 2 and 4)
        num_attractions_to_sample = int(self.rng.choice([2, 3, 4], size=1)[0])

        selected_attractions, selected_distances = self._weighted_regional_sample(
            attraction_distances, n=num_attractions_to_sample
        )

        # Calculate distance for each possible route
        routes = []
        for perm in permutations(selected_attractions):
            total_distance = 0
            current_location = hotel.location
            for attraction in perm:
                total_distance += distance_meters(current_location, attraction.location)
                current_location = attraction.location
            routes.append({
                'order': [attr.name for attr in perm],
                'total_distance': total_distance,
            })

        optimal_route = min(routes, key=lambda x: x['total_distance'])
        optimal_order = optimal_route['order']
        optimal_distance = optimal_route['total_distance']

        hotels_set_json = json.dumps([h.to_dict() for h in hotels_set])

        units_info = " The locations are given in meters from the origin, a (0, 0) point in the southwest corner of the map."

        attraction_names = [capitalize_name(attr.name) for attr in selected_attractions]
        if num_attractions_to_sample == 2:
            attractions_list_str = f"{attraction_names[0]} and {attraction_names[1]}"
            them_word = "them"
        else:
            attractions_list_str = ", ".join(attraction_names[:-1]) + f", and {attraction_names[-1]}"
            them_word = "all of them"

        user_question = f"I want to visit {attractions_list_str}. I'll walk between them starting from the hotel and taking a taxi back from the last one. What order should I visit {them_word} in to minimize my walking distance?"

        tool_call_hotel, tool_call_hotel_id = self._make_tool_call(
            "search_hotels", {"pricerange": hotel_pricerange, "area": hotel_area})
        tool_response_hotel = self._make_tool_response(tool_call_hotel_id, hotels_set_json)

        current_date = self._random_date()
        current_time = self._random_time()
        num_nights = int(self.rng.integers(2, 7))
        checkin_date = current_date + datetime.timedelta(days=1)
        checkout_date = current_date + datetime.timedelta(days=num_nights + 1)

        tool_call_booking, tool_call_booking_id = self._make_tool_call(
            "book_hotel", {
                "hotel_id": hotel.id,
                "rooms": [{
                    "checkin_date": checkin_date.isoformat(),
                    "checkout_date": checkout_date.isoformat(),
                    "room_type": "double",
                    "num_guests": 2,
                }],
            })
        tool_response_booking = self._make_tool_response(tool_call_booking_id, "Hotel booked successfully")

        hotel_area_adj = self._area_adj(hotel_area)
        hotel_pricerange_word = self._format_pricerange(hotel_pricerange)
        stars_phrase = self._stars_phrase(hotel.stars)

        placeholders = {
            "tool_call_hotel": tool_call_hotel,
            "tool_response_hotel": tool_response_hotel,
            "tool_call_booking": tool_call_booking,
            "tool_response_booking": tool_response_booking,
        }
        template_vars = {
            "hotel_pricerange": hotel_pricerange_word,
            "hotel_area_adj": hotel_area_adj,
            "hotel_name": capitalize_name(hotel.name),
            "stars_phrase": stars_phrase,
            "hotel_type": hotel.type,
            "hotel_address": capitalize_name(hotel.address),
            "num_nights": number_to_word(num_nights),
        }
        rendered_messages = self._render_dialogue(dialogue_template, placeholders, template_vars)

        self._append_attraction_queries(selected_attractions, rendered_messages)
        rendered_messages.append({"role": "user", "content": user_question})

        attractions_set_json = json.dumps([a.to_dict() for a in selected_attractions])
        prompt = f"Given the hotels and attractions in JSON format below, determine the optimal order to walk between {attractions_list_str} starting from {capitalize_name(hotel.name)}. The user will take a taxi back from the last one, and the total walking distance should be minimized.{units_info}\n\nHotels:\n{hotels_set_json}\n\nAttractions:\n{attractions_set_json}"

        distance_details: dict[str, dict | float] = {
            'hotel_to_attractions': {
                attr.name: distance_meters(hotel.location, attr.location)
                for attr in selected_attractions
            },
            **{
                f"{a1.name}_to_{a2.name}": distance_meters(a1.location, a2.location)
                for a1 in selected_attractions
                for a2 in selected_attractions
                if a1 is not a2
            },
        }

        return {
            "messages": [{"role": "system", "content": prompt}] + rendered_messages,
            "current_date": current_date.isoformat(),
            "current_time": current_time.isoformat("minutes"),
            "hotel": hotel.to_dict(),
            "hotel_params": {
                "area": hotel_area,
                "pricerange": hotel_pricerange,
            },
            "num_attractions_sampled": num_attractions_to_sample,
            "selected_attractions": [attr.to_dict() for attr in selected_attractions],
            "targets": {
                "optimal_route_order": optimal_order,
                "optimal_distance": optimal_distance,
                "all_routes": routes,
                "distance_details": distance_details,
            },
        }

    def _append_attraction_queries(self, selected_attractions: list,
                                   messages: list[dict]) -> None:
        for i, attr in enumerate(selected_attractions):
            attr_name = capitalize_name(attr.name)

            if attr.openhours not in ["?", "always", None]:
                openhours = " " + attr.openhours.capitalize().strip(".") + "."
            elif attr.openhours == "always":
                openhours = " It is open 24/7."
            else:
                openhours = ""

            if attr.entrance_fee not in ["?", "free", None]:
                entrance_fee = " The entrance fee is " + attr.entrance_fee + "."
            elif attr.entrance_fee == "free":
                entrance_fee = " The entrance is free of charge."
            else:
                entrance_fee = ""

            templates = self.USER_REQUEST_TEMPLATES_FIRST if i == 0 else self.USER_REQUEST_TEMPLATES_ADDITIONAL
            user_request = self.rng.choice(templates).format(attr_name=attr_name)

            tool_call, tool_call_id = self._make_tool_call(
                "search_attractions", {"name": attr.name})
            tool_response = self._make_tool_response(
                tool_call_id, json.dumps(attr.to_dict()))

            area_adj = self._area_adj(attr.area.value)
            assistant_response = f"{attr_name} is a {attr.type} located at {capitalize_name(attr.address)} in the {area_adj} part of town.{openhours}{entrance_fee}"

            messages.extend([
                {"role": "user", "content": user_request},
                tool_call,
                tool_response,
                {"role": "assistant", "content": assistant_response},
            ])


class HotelPriceGenerator(BenchmarkGenerator):

    def generate(self, dialogue_template: list[dict] | None = None) -> dict:
        hotel_db = self._load_db("hotel_db")

        current_date = self._random_date()
        current_time = self._random_time()

        # Select a random hotel
        hotels = hotel_db.query()
        hotels_with_prices = [h for h in hotels if h.price and (h.price.single or h.price.double or h.price.family)]
        hotel = self.rng.choice(hotels_with_prices, size=1, replace=False)[0]

        num_people = int(self.rng.integers(2, 7))
        num_nights = int(self.rng.integers(2, 8))

        rooms = self._allocate_rooms(num_people, hotel)
        exceptions = self._generate_exceptions(num_people, rooms)
        total_cost = self._calculate_total_cost(rooms, exceptions, num_nights, hotel)

        hotel_params = {
            "area": hotel.area.value,
            "hotel_type": hotel.type,
            "pricerange": hotel.pricerange.value,
        }

        hotels_matching = hotel_db.query(**{k: v for k, v in hotel_params.items() if v is not None})
        hotels_matching_json = json.dumps([h.to_dict() for h in hotels_matching])

        room_counts = Counter(room["type"] for room in rooms)
        room_allocation_str, has_mixed_room_types = self._format_room_allocation(
            rooms, num_people, room_counts)
        exception_str = self._format_exceptions(exceptions, rooms, room_counts)

        # Build query parameters description
        query_parts = []
        if hotel_params["pricerange"]:
            query_parts.append(self._format_pricerange(hotel_params["pricerange"]))
        if hotel_params["hotel_type"]:
            query_parts.append(hotel_params['hotel_type'])
        if hotel_params["area"]:
            query_parts.append(f"in the {self._area_adj(hotel_params['area'])} part of town")
        query_desc = " ".join(query_parts) if query_parts else "hotels"

        stars_phrase = self._stars_phrase(hotel.stars)
        hotel_formatted = f"{capitalize_name(hotel.name)}, a {stars_phrase}{hotel.type} with great reviews located at {capitalize_name(hotel.address)}"

        # Format user message and prompt room description
        if has_mixed_room_types:
            room_desc_cap = room_allocation_str[0].upper() + room_allocation_str[1:] if room_allocation_str else ""
            user_message = f"Can you first calculate the total price for me? It's for {number_to_word(num_people)} people staying for {number_to_word(num_nights)} nights. {room_desc_cap}.{exception_str} How much would it cost?"
            prompt_room_desc = room_desc_cap
        else:
            user_message = f"Can you first calculate the total price for me? It's for {number_to_word(num_people)} people staying for {number_to_word(num_nights)} nights {room_allocation_str}.{exception_str} How much would it cost?"
            prompt_room_desc = f"in {room_allocation_str}" if " in " not in room_allocation_str and not room_allocation_str.startswith("in ") else room_allocation_str

        tool_call_hotel, tool_call_hotel_id = self._make_tool_call(
            "search_hotels", {k: v for k, v in hotel_params.items() if v is not None})
        tool_response_hotel = self._make_tool_response(tool_call_hotel_id, hotels_matching_json)

        placeholders = {
            "tool_call_hotel": tool_call_hotel,
            "tool_response_hotel": tool_response_hotel,
        }
        template_vars = {
            "query_desc": query_desc,
            "hotel_formatted": hotel_formatted,
            "user_message": user_message,
        }
        rendered_messages = self._render_dialogue(dialogue_template, placeholders, template_vars)

        if has_mixed_room_types:
            prompt = f"Given the hotel information in JSON format below, calculate the total cost for booking {capitalize_name(hotel.name)} for {number_to_word(num_nights)} nights for {number_to_word(num_people)} people. {prompt_room_desc}.{exception_str}\n\nHotels:\n{hotels_matching_json}"
        else:
            prompt = f"Given the hotel information in JSON format below, calculate the total cost for booking {capitalize_name(hotel.name)} for {number_to_word(num_nights)} nights for {number_to_word(num_people)} people {prompt_room_desc}.{exception_str}\n\nHotels:\n{hotels_matching_json}"

        return {
            "messages": [{"role": "system", "content": prompt}] + rendered_messages,
            "current_date": current_date.isoformat(),
            "current_time": current_time.isoformat("minutes"),
            "hotel_params": hotel_params,
            "hotel": hotel.to_dict(),
            "hotels_all": [h.to_dict() for h in hotels_matching],
            "num_people": num_people,
            "num_nights": num_nights,
            "rooms": rooms,
            "exceptions": exceptions,
            "total_price": total_cost,
        }

    def _allocate_rooms(self, num_people: int, hotel) -> list[dict]:
        remaining_people = num_people
        rooms: list[dict] = []

        available_room_types: list[tuple[str, int]] = []
        if hotel.price.family is not None:
            available_room_types.append(("family", 4))
        if hotel.price.double is not None:
            available_room_types.append(("double", 2))
        if hotel.price.single is not None:
            available_room_types.append(("single", 1))

        while remaining_people > 0:
            valid_room_types = [(room_type, capacity) for room_type, capacity in available_room_types
                               if capacity <= remaining_people]

            if not valid_room_types:
                capacity_by_type = dict(available_room_types)
                allocated = False
                for room in rooms:
                    cap = capacity_by_type.get(room["type"], 0)
                    if room["people"] < cap:
                        can_add = min(cap - room["people"], remaining_people)
                        room["people"] += can_add
                        remaining_people -= can_add
                        allocated = True
                        break

                if not allocated:
                    if available_room_types:
                        room_type, capacity = available_room_types[0]
                        rooms.append({"type": room_type, "people": remaining_people})
                        remaining_people = 0
                    break

            selected_idx = int(self.rng.integers(0, len(valid_room_types)))
            room_type, capacity = valid_room_types[selected_idx]
            rooms.append({"type": room_type, "people": capacity})
            remaining_people -= capacity

        return rooms

    def _generate_exceptions(self, num_people: int, rooms: list[dict]) -> list[dict]:
        max_exceptions = max(1, num_people // 2 - 1) if num_people > 2 else 0
        num_exceptions = int(self.rng.integers(0, max_exceptions + 1)) if max_exceptions > 0 else 0

        if num_exceptions == 0:
            return []

        exception_types = [
            ("check-out", ["check out one day earlier", "check out one day later"]),
            ("check-in", ["check in one day earlier", "check in one day later"]),
        ]

        exceptions = []
        exception_people = self.rng.choice(num_people, size=num_exceptions, replace=False)

        for person_idx in exception_people:
            person_count = 0
            room_idx = 0
            for i, room in enumerate(rooms):
                if person_count <= person_idx < person_count + room["people"]:
                    room_idx = i
                    break
                person_count += room["people"]

            exception_type_idx = int(self.rng.integers(0, len(exception_types)))
            exception_category, exception_phrases = exception_types[exception_type_idx]
            exception_phrase_idx = int(self.rng.integers(0, len(exception_phrases)))
            exception_phrase = exception_phrases[exception_phrase_idx]

            if "earlier" in exception_phrase or "early" in exception_phrase:
                nights_change = -1 if exception_category == "check-out" else 1
            else:
                nights_change = 1 if exception_category == "check-out" else -1

            exceptions.append({
                "person_idx": int(person_idx),
                "room_idx": room_idx,
                "category": exception_category,
                "phrase": exception_phrase,
                "nights_change": nights_change,
            })

        return exceptions

    def _calculate_total_cost(self, rooms: list[dict], exceptions: list[dict],
                              num_nights: int, hotel) -> float:
        room_nights_needed = []
        for room_idx, room in enumerate(rooms):
            person_count = 0
            people_in_room: list[int] = []
            for i, r in enumerate(rooms):
                if i < room_idx:
                    person_count += r["people"]
                elif i == room_idx:
                    people_in_room = list(range(person_count, person_count + r["people"]))
                    break

            person_schedules = {
                pid: {"checkin": 0, "checkout": num_nights}
                for pid in people_in_room
            }

            for exception in exceptions:
                if exception["person_idx"] in people_in_room:
                    if exception["category"] == "check-in":
                        person_schedules[exception["person_idx"]]["checkin"] -= exception["nights_change"]
                    else:
                        person_schedules[exception["person_idx"]]["checkout"] += exception["nights_change"]

            earliest_checkin = min(s["checkin"] for s in person_schedules.values())
            latest_checkout = max(s["checkout"] for s in person_schedules.values())
            room_nights_needed.append(latest_checkout - earliest_checkin)

        total_cost = 0.0
        for room_idx, room in enumerate(rooms):
            room_price = float(getattr(hotel.price, room["type"]))
            total_cost += room_price * room_nights_needed[room_idx]

        return total_cost

    def _format_room_allocation(self, rooms: list[dict], num_people: int,
                                room_counts: Counter) -> tuple[str, bool]:
        has_mixed_room_types = len(room_counts) > 1

        if (len(room_counts) == 1 and "single" in room_counts
                and room_counts["single"] == num_people):
            return "each in their own single room", False

        if has_mixed_room_types:
            room_parts = []

            room_type_people: dict[str, int] = defaultdict(int)
            for room in rooms:
                room_type_people[room["type"]] += room["people"]

            sorted_room_types = sorted(
                room_counts.keys(),
                key=lambda rt: (room_counts[rt], room_type_people[rt])
            )

            for idx, room_type in enumerate(sorted_room_types):
                count = room_counts[room_type]
                total_people = room_type_people[room_type]
                people_count = next(r["people"] for r in rooms if r["type"] == room_type)

                is_last = idx == len(sorted_room_types) - 1
                use_others = is_last and len(room_counts) == 2

                if count == 1 and room_type in ["family", "double"] and people_count > 1:
                    if use_others:
                        room_parts.append(f"the others will share a {room_type} room")
                    else:
                        room_parts.append(f"{number_to_word(people_count)} people will share a {room_type} room")
                elif count > 1 and room_type in ["family", "double"]:
                    if use_others:
                        room_parts.append(f"the others will share {room_type} rooms")
                    else:
                        room_parts.append(f"{number_to_word(total_people)} people will share {room_type} rooms")
                else:
                    if count == 1:
                        if people_count == 1:
                            if use_others:
                                room_parts.append(f"the other will stay in a {room_type} room")
                            else:
                                room_parts.append(f"one person will stay in a {room_type} room")
                        else:
                            if use_others:
                                room_parts.append(f"the others will stay in a {room_type} room")
                            else:
                                room_parts.append(f"{number_to_word(people_count)} people will stay in a {room_type} room")
                    else:
                        if room_type == "single" and total_people >= 2:
                            if use_others:
                                room_parts.append(f"the others will each stay in their own single room")
                            else:
                                room_parts.append(f"{number_to_word(total_people)} people will each stay in their own single room")
                        else:
                            if use_others:
                                room_parts.append(f"the others will stay in {room_type} rooms")
                            else:
                                room_parts.append(f"{number_to_word(total_people)} people will stay in {room_type} rooms")

            return " and ".join(room_parts), True

        room_parts = []
        for room_type, count in room_counts.items():
            if count == 1:
                room_parts.append(f"a {room_type} room")
            else:
                room_parts.append(f"{number_to_word(count)} {room_type} rooms")

        if len(room_parts) == 1:
            alloc = room_parts[0]
        elif len(room_parts) == 2:
            alloc = f"{room_parts[0]} and {room_parts[1]}"
        else:
            alloc = ", ".join(room_parts[:-1]) + f", and {room_parts[-1]}"
        return f"in {alloc}", False

    def _format_exceptions(self, exceptions: list[dict], rooms: list[dict],
                           room_counts: Counter) -> str:
        if not exceptions:
            return ""

        exception_groups: Counter[tuple[str, str | None]] = Counter()
        for exception in exceptions:
            if len(room_counts) == 1:
                key = (exception['phrase'], None)
            else:
                room_type = rooms[exception["room_idx"]]["type"]
                key = (exception['phrase'], room_type)
            exception_groups[key] += 1

        exception_parts = []
        for (phrase, room_type), count in exception_groups.items():
            person_str = "one person" if count == 1 else f"{number_to_word(count)} people"

            if room_type is None:
                exception_parts.append(f"{person_str} will {phrase}")
            else:
                room_str = f"{room_type} rooms" if count > 1 else f"a {room_type} room"
                exception_parts.append(f"{person_str} in {room_str} will {phrase}")

        if len(exception_parts) == 1:
            return " " + exception_parts[0].capitalize() + "."
        elif len(exception_parts) == 2:
            return " " + exception_parts[0].capitalize() + " and " + exception_parts[1] + "."
        else:
            return " " + exception_parts[0].capitalize() + ", " + ", ".join(exception_parts[1:-1]) + ", and " + exception_parts[-1] + "."


def select_article_for_pricerange(word: str) -> str:
    return "an" if word == "expensive" else "a"


def is_direction_correct(asked_direction: str, location1: tuple, location2: tuple) -> bool:
    dx = location2[0] - location1[0]
    dy = location2[1] - location1[1]

    if asked_direction == 'north':
        return dy > 0
    elif asked_direction == 'south':
        return dy < 0
    elif asked_direction == 'east':
        return dx > 0
    elif asked_direction == 'west':
        return dx < 0
    else:
        raise ValueError(f"Invalid direction: {asked_direction}")


def capitalize_name(name: str) -> str:
    return ' '.join([word.capitalize() for word in name.split()])


def distance_meters(location1: tuple, location2: tuple) -> float:
    return math.sqrt((location1[0] - location2[0])**2 + (location1[1] - location2[1])**2)
