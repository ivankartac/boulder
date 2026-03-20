import datetime
import json
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class Area(Enum):
    CENTRE = "centre"
    EAST = "east"
    WEST = "west"
    NORTH = "north"
    SOUTH = "south"


class PriceRange(Enum):
    FREE = "free"
    CHEAP = "cheap"
    MODERATE = "moderate"
    EXPENSIVE = "expensive"
    UNKNOWN = "?"


class AttractionType(Enum):
    ARCHITECTURE = "architecture"
    BOAT = "boat"
    CINEMA = "cinema"
    COLLEGE = "college"
    CONCERTHALL = "concerthall"
    ENTERTAINMENT = "entertainment"
    MULTIPLE_SPORTS = "mutliple sports"
    MUSEUM = "museum"
    NIGHTCLUB = "nightclub"
    PARK = "park"
    SWIMMINGPOOL = "swimmingpool"
    THEATRE = "theatre"


class HotelType(Enum):
    HOTEL = "hotel"
    GUESTHOUSE = "guesthouse"


class RoomType(Enum):
    SINGLE = "single"
    DOUBLE = "double"
    FAMILY = "family"


class FoodType(Enum):
    AFRICAN = "african"
    ASIAN_ORIENTAL = "asian oriental"
    BRITISH = "british"
    CHINESE = "chinese"
    EUROPEAN = "european"
    FRENCH = "french"
    GASTROPUB = "gastropub"
    INDIAN = "indian"
    INTERNATIONAL = "international"
    ITALIAN = "italian"
    JAPANESE = "japanese"
    KOREAN = "korean"
    LEBANESE = "lebanese"
    MEDITERRANEAN = "mediterranean"
    MEXICAN = "mexican"
    MODERN_EUROPEAN = "modern european"
    NORTH_AMERICAN = "north american"
    PORTUGUESE = "portuguese"
    SEAFOOD = "seafood"
    SPANISH = "spanish"
    THAI = "thai"
    TURKISH = "turkish"
    VIETNAMESE = "vietnamese"


class Weekday(Enum):
    MONDAY = "monday"
    TUESDAY = "tuesday"
    WEDNESDAY = "wednesday"
    THURSDAY = "thursday"
    FRIDAY = "friday"
    SATURDAY = "saturday"
    SUNDAY = "sunday"


class Station(Enum):
    BIRMINGHAM_NEW_STREET = "birmingham new street"
    BISHOPS_STORTFORD = "bishops stortford"
    BROXBOURNE = "broxbourne"
    CAMBRIDGE = "cambridge"
    ELY = "ely"
    KINGS_LYNN = "kings lynn"
    LEICESTER = "leicester"
    LONDON_KINGS_CROSS = "london kings cross"
    LONDON_LIVERPOOL_STREET = "london liverpool street"
    NORWICH = "norwich"
    PETERBOROUGH = "peterborough"
    STANSTED_AIRPORT = "stansted airport"
    STEVENAGE = "stevenage"


@dataclass
class Attraction:
    id: str
    name: str
    address: str
    area: Area
    entrance_fee: str
    location: tuple[float, float]
    openhours: str
    phone: str
    postcode: str
    pricerange: PriceRange
    type: str

    @classmethod
    def from_dict(cls, data: dict) -> 'Attraction':
        return cls(
            id=data['id'],
            name=data['name'],
            address=data['address'],
            area=Area(data['area']),
            entrance_fee=data['entrance fee'],
            location=(data['location'][0], data['location'][1]),
            openhours=data.get('openhours'),
            phone=data['phone'],
            postcode=data['postcode'],
            pricerange=PriceRange(data['pricerange']),
            type=data['type']
        )

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'address': self.address,
            'area': self.area.value,
            'entrance fee': self.entrance_fee,
            'location': list(self.location),
            'openhours': self.openhours,
            'phone': self.phone,
            'postcode': self.postcode,
            'pricerange': self.pricerange.value,
            'type': self.type
        }


@dataclass
class HotelPrice:
    single: str | None = None
    double: str | None = None
    family: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> 'HotelPrice':
        return cls(
            single=data.get('single'),
            double=data.get('double'),
            family=data.get('family')
        )

    def to_dict(self) -> dict:
        result = {}
        if self.single is not None:
            result['single'] = self.single
        if self.double is not None:
            result['double'] = self.double
        if self.family is not None:
            result['family'] = self.family
        return result


@dataclass
class Hotel:
    id: str
    name: str
    address: str
    area: Area
    internet: str
    parking: str
    location: tuple[float, float]
    phone: str
    postcode: str
    price: HotelPrice
    pricerange: PriceRange
    stars: str
    takesbookings: str
    type: str

    @classmethod
    def from_dict(cls, data: dict) -> 'Hotel':
        return cls(
            id=data['id'],
            name=data['name'],
            address=data['address'],
            area=Area(data['area']),
            internet=data['internet'],
            parking=data['parking'],
            location=(data['location'][0], data['location'][1]),
            phone=data['phone'],
            postcode=data['postcode'],
            price=HotelPrice.from_dict(data['price']),
            pricerange=PriceRange(data['pricerange']),
            stars=data['stars'],
            takesbookings=data['takesbookings'],
            type=data['type']
        )

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'address': self.address,
            'area': self.area.value,
            'internet': self.internet,
            'parking': self.parking,
            'location': list(self.location),
            'phone': self.phone,
            'postcode': self.postcode,
            'price': self.price.to_dict(),
            'pricerange': self.pricerange.value,
            'stars': self.stars,
            'takesbookings': self.takesbookings,
            'type': self.type
        }


@dataclass
class Restaurant:
    id: str
    name: str
    address: str
    area: Area
    food: str
    location: tuple[float, float]
    phone: str
    postcode: str
    pricerange: PriceRange
    type: str
    openhours: dict[str, dict[str, str]] | None = None
    introduction: str | None = None
    signature: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> 'Restaurant':
        def safe_get_str(key: str) -> str | None:
            value = data.get(key)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return None
            return str(value) if value else None

        return cls(
            id=data['id'],
            name=data['name'],
            address=data['address'],
            area=Area(data['area']),
            food=data['food'],
            location=(data['location'][0], data['location'][1]),
            phone=safe_get_str('phone') or '',
            postcode=data['postcode'],
            pricerange=PriceRange(data['pricerange']),
            openhours=data.get('openhours'),
            type=data['type'],
            introduction=safe_get_str('introduction'),
            signature=safe_get_str('signature')
        )

    def to_dict(self) -> dict:
        result = {
            'id': self.id,
            'name': self.name,
            'address': self.address,
            'area': self.area.value,
            'food': self.food,
            'location': list(self.location),
            'phone': self.phone,
            'postcode': self.postcode,
            'pricerange': self.pricerange.value,
            'openhours': self.openhours,
            'type': self.type
        }

        if self.introduction is not None:
            result['introduction'] = self.introduction
        if self.signature is not None:
            result['signature'] = self.signature

        return result


class BaseModelDB(ABC):
    def __init__(self, data: dict):
        self.df = pd.DataFrame.from_dict(data)

    @classmethod
    def from_json(cls, json_file_path: str) -> type['BaseModelDB']:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        return cls(data)

    @property
    @abstractmethod
    def model(self) -> type[object]:
        pass

    @abstractmethod
    def query(self, **kwargs) -> list[object]:
        pass

    def _df_to_objects(self, df: pd.DataFrame) -> list[object]:
        objects = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            obj = self.model.from_dict(row_dict)
            objects.append(obj)
        return objects

    def get_schema(self) -> dict:
        schema = {}
        for field in self.df.columns:
            try:
                enum = self.df[field].unique().tolist() if field != "id" else None
            except TypeError:
                enum = None
            schema[field] = {
                "type": "string",
                "enum": enum
            }
        return schema


class AttractionDB(BaseModelDB):
    @property
    def model(self) -> type[Attraction]:
        return Attraction

    def query(
        self,
        name: str = None,
        area: str | Area = None,
        attraction_type: str = None,
        pricerange: str | PriceRange = None,
    ) -> list[Attraction]:
        filtered_df = self.df.copy()

        if name:
            filtered_df = filtered_df[
                filtered_df['name'].str.contains(name, case=False, na=False)
            ]

        if area:
            area_value = area.value if isinstance(area, Area) else area
            filtered_df = filtered_df[filtered_df['area'] == area_value]

        if attraction_type:
            filtered_df = filtered_df[filtered_df['type'] == attraction_type]

        if pricerange:
            price_value = pricerange.value if isinstance(pricerange, PriceRange) else pricerange
            filtered_df = filtered_df[filtered_df['pricerange'] == price_value]

        return self._df_to_objects(filtered_df)


class HotelDB(BaseModelDB):
    @property
    def model(self) -> type[Hotel]:
        return Hotel

    def query(
        self,
        name: str = None,
        area: str | Area = None,
        hotel_type: str = None,
        pricerange: str | PriceRange = None,
        stars: int = None,
        min_stars: int = None,
        max_price: int = None,
        room_type: str = None,
    ) -> list[Hotel]:
        filtered_df = self.df.copy()

        if name:
            filtered_df = filtered_df[
                filtered_df['name'].str.contains(name, case=False, na=False)
            ]

        if area:
            area_value = area.value if isinstance(area, Area) else area
            filtered_df = filtered_df[filtered_df['area'] == area_value]

        if hotel_type:
            filtered_df = filtered_df[filtered_df['type'] == hotel_type]

        if pricerange:
            price_value = pricerange.value if isinstance(pricerange, PriceRange) else pricerange
            filtered_df = filtered_df[filtered_df['pricerange'] == price_value]

        if stars is not None:
            filtered_df = filtered_df[filtered_df['stars'] == str(stars)]

        if min_stars is not None:
            numeric_stars = pd.to_numeric(filtered_df['stars'], errors='coerce')
            filtered_df = filtered_df[numeric_stars >= min_stars]

        if max_price is not None:
            def has_affordable_room(row):
                prices = row['price']
                if room_type is not None:
                    room_prices = [prices.get(room_type.lower())]
                else:
                    room_prices = [prices.get('single'), prices.get('double'), prices.get('family')]

                for room_price in room_prices:
                    if room_price is not None:
                        try:
                            if int(room_price) <= max_price:
                                return True
                        except ValueError:
                            continue
                return False

            filtered_df = filtered_df[filtered_df.apply(has_affordable_room, axis=1)]

        if room_type:
            def has_room_type(row):
                prices = row['price']
                return prices.get(room_type.lower()) is not None

            filtered_df = filtered_df[filtered_df.apply(has_room_type, axis=1)]

        return self._df_to_objects(filtered_df)


class RestaurantDB(BaseModelDB):
    @property
    def model(self) -> type[Restaurant]:
        return Restaurant

    def query(
        self,
        name: str = None,
        area: str | Area = None,
        food: str | FoodType = None,
        pricerange: str | PriceRange = None,
    ) -> list[Restaurant]:
        filtered_df = self.df.copy()

        if name:
            filtered_df = filtered_df[
                filtered_df['name'].str.contains(name, case=False, na=False)
            ]

        if area:
            area_value = area.value if isinstance(area, Area) else area
            filtered_df = filtered_df[filtered_df['area'] == area_value]

        if food:
            food_value = food.value if isinstance(food, FoodType) else food
            filtered_df = filtered_df[filtered_df['food'] == food_value]

        if pricerange:
            price_value = pricerange.value if isinstance(pricerange, PriceRange) else pricerange
            filtered_df = filtered_df[filtered_df['pricerange'] == price_value]

        return self._df_to_objects(filtered_df)


@dataclass
class Train:
    train_id: str
    departure: str
    destination: str
    day: Weekday
    leave_at: datetime.time
    arrive_by: datetime.time
    duration: str
    price_standard: str
    price_first_class: str | None = None
    metadata: dict | None = None

    @classmethod
    def from_dict(cls, data: dict) -> 'Train':
        arrive_by = data["arriveBy"]
        if "24:" in arrive_by:
            arrive_by = arrive_by.replace("24:", "00:")

        price_standard = data.get('price_standard') or data.get('price')
        price_first_class = data.get('price_first_class')

        return cls(
            train_id=data['trainID'],
            departure=data['departure'],
            destination=data['destination'],
            day=Weekday(data['day']),
            leave_at=datetime.time.fromisoformat(data['leaveAt']),
            arrive_by=datetime.time.fromisoformat(arrive_by),
            duration=data['duration'],
            price_standard=price_standard,
            price_first_class=price_first_class,
            metadata=data.get('metadata')
        )

    def to_dict(self) -> dict:
        return {
            'trainID': self.train_id,
            'departure': self.departure,
            'destination': self.destination,
            'day': self.day.value,
            'leaveAt': self.leave_at.isoformat("minutes"),
            'arriveBy': self.arrive_by.isoformat("minutes"),
            'duration': self.duration,
            'price_standard': self.price_standard,
            'price_first_class': self.price_first_class,
        }

    def get_price_as_float(self, is_first_class: bool = False) -> float | None:
        try:
            if is_first_class:
                price_field = self.price_first_class
            else:
                price_field = self.price_standard
            price_str = price_field.replace('pounds', '').strip()
            return float(price_str)
        except (ValueError, AttributeError):
            return None


class TrainDB(BaseModelDB):
    @property
    def model(self) -> type[Train]:
        return Train

    def query(
        self,
        departure: str = None,
        destination: str = None,
        weekday: str | Weekday = None,
        leave_before: str = None,
        leave_after: str = None,
        arrive_after: str = None,
        arrive_before: str = None,
    ) -> list[Train]:
        filtered_df = self.df.copy()

        if departure:
            filtered_df = filtered_df[
                filtered_df['departure'].str.contains(departure, case=False, na=False)
            ]

        if destination:
            filtered_df = filtered_df[
                filtered_df['destination'].str.contains(destination, case=False, na=False)
            ]

        if weekday:
            day_value = weekday.value if isinstance(weekday, Weekday) else weekday
            filtered_df = filtered_df[filtered_df['day'] == day_value]

        if leave_after:
            leave_after_object = datetime.time.fromisoformat(leave_after)
            def leaves_after(row):
                try:
                    leave_time = row['leaveAt']
                    leave_time = datetime.time.fromisoformat(leave_time)
                    return leave_time >= leave_after_object
                except (ValueError, TypeError):
                    return False
            filtered_df = filtered_df[filtered_df.apply(leaves_after, axis=1)]

        if leave_before:
            leave_before_object = datetime.time.fromisoformat(leave_before)
            def leaves_before(row):
                try:
                    leave_time = row['leaveAt']
                    leave_time = datetime.time.fromisoformat(leave_time)
                    return leave_time <= leave_before_object
                except (ValueError, TypeError):
                    return False
            filtered_df = filtered_df[filtered_df.apply(leaves_before, axis=1)]

        if arrive_before:
            arrive_before_object = datetime.time.fromisoformat(arrive_before)
            def arrives_before(row):
                try:
                    arrive_time = row['arriveBy']
                    arrive_time = datetime.time.fromisoformat(arrive_time)
                    return arrive_time <= arrive_before_object
                except (ValueError, TypeError):
                    return False
            filtered_df = filtered_df[filtered_df.apply(arrives_before, axis=1)]

        if arrive_after:
            arrive_after_object = datetime.time.fromisoformat(arrive_after)
            def arrives_after(row):
                try:
                    arrive_time = row['arriveBy']
                    arrive_time = datetime.time.fromisoformat(arrive_time)
                    return arrive_time >= arrive_after_object
                except (ValueError, TypeError):
                    return False
            filtered_df = filtered_df[filtered_df.apply(arrives_after, axis=1)]

        return self._df_to_objects(filtered_df)
