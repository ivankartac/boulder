import datetime
import pydantic
from boulder.domains.db import AttractionDB, HotelDB, AttractionType, Area, PriceRange, TrainDB, RestaurantDB, Weekday, FoodType, HotelType, RoomType, Station


class BaseTool:
    def __init__(self, name: str, description: str, parameters: pydantic.BaseModel):
        self.name = name
        self.description = description
        self.parameters = parameters

    def get_tool_schema(self) -> dict:
        parameters = self._filter_schema_fields(self.parameters.model_json_schema())
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    def _filter_schema_fields(self, parameters: dict) -> dict:
        if "title" in parameters:
            del parameters["title"]
        if "properties" in parameters:
            for value in parameters["properties"].values():
                if "title" in value:
                    del value["title"]
        return parameters


class BaseSearchTool(BaseTool):
    def __init__(self, db, name: str, description: str, parameters: pydantic.BaseModel):
        super().__init__(name, description, parameters)
        self.db = db

    def __call__(self, parameters: pydantic.BaseModel) -> list[dict]:
        result = self.db.query(**parameters.model_dump())
        return [r.to_dict() for r in result]


class SearchRestaurantsToolParameters(pydantic.BaseModel):
    area: str = pydantic.Field(
        None,
        description="The area to search for restaurants in",
        enum=[x.value for x in Area],
    )
    food: str = pydantic.Field(
        None,
        description="The type of food to search for",
        enum=[x.value for x in FoodType],
    )
    pricerange: str = pydantic.Field(
        None,
        description="The price range to search for",
        enum=[x.value for x in PriceRange if x.value != PriceRange.UNKNOWN.value],
    )


class SearchRestaurantsTool(BaseSearchTool):
    def __init__(
        self,
        db: RestaurantDB,
        name: str = "search_restaurants",
        description: str = "Search for restaurants with the given parameters",
    ):
        super().__init__(db, name, description, SearchRestaurantsToolParameters)

    def __call__(self, parameters: SearchRestaurantsToolParameters) -> list[dict]:
        parameters.food = parameters.food.lower() if parameters.food else None
        parameters.area = parameters.area.lower() if parameters.area else None
        parameters.pricerange = parameters.pricerange.lower() if parameters.pricerange else None
        return super().__call__(parameters)


class RestaurantReservationToolParameters(pydantic.BaseModel):
    restaurant_id: str = pydantic.Field(
        ...,
        description="The ID of the restaurant to make a reservation at",
    )
    date: datetime.date = pydantic.Field(
        ...,
        description="The date of the reservation (YYYY-MM-DD format)",
    )
    time: datetime.time = pydantic.Field(
        ...,
        description="The time of the reservation (HH:MM format)",
    )
    num_guests: int = pydantic.Field(
        ...,
        description="The number of guests to make a reservation for",
    )


class RestaurantReservationTool(BaseTool):
    def __init__(
        self,
        db: RestaurantDB,
        name: str = "make_restaurant_reservation",
        description: str = "Make a reservation at a specific restaurant",
    ):
        super().__init__(name, description, RestaurantReservationToolParameters)
        self.db = db

    def __call__(self, parameters: RestaurantReservationToolParameters) -> dict:
        return {"success": True, "message": "Restaurant reservation successful"}


class SearchAttractionsToolParameters(pydantic.BaseModel):
    name: str = pydantic.Field(
        None,
        description="The name of the attraction to search for if looking for a specific attraction",
    )
    area: str = pydantic.Field(
        None,
        description="The area to search for attractions in",
        enum=[x.value for x in Area],
    )
    attraction_type: str = pydantic.Field(
        None,
        description="The type of the attraction to search for",
        enum=[x.value for x in AttractionType],
    )
    pricerange: str = pydantic.Field(
        None,
        description="The price range to search for",
        enum=[x.value for x in PriceRange if x.value != PriceRange.UNKNOWN.value],
    )


class SearchAttractionsTool(BaseSearchTool):
    def __init__(
        self,
        db: AttractionDB,
        name: str = "search_attractions",
        description: str = "Search for attractions with the given parameters",
    ):
        super().__init__(db, name, description, SearchAttractionsToolParameters)


class SearchHotelsToolParameters(pydantic.BaseModel):
    area: str = pydantic.Field(
        None,
        description="The area to search for hotels in",
        enum=[x.value for x in Area],
    )
    hotel_type: str = pydantic.Field(
        None,
        description="The type of the hotel to search for",
        enum=[x.value for x in HotelType],
    )
    pricerange: str = pydantic.Field(
        None,
        description="The price range to search for",
        enum=[x.value for x in PriceRange if x.value != PriceRange.UNKNOWN.value],
    )


class SearchHotelsTool(BaseSearchTool):
    def __init__(
        self,
        db: HotelDB,
        name: str = "search_hotels",
        description: str = "Search for hotels with the given parameters",
    ):
        super().__init__(db, name, description, SearchHotelsToolParameters)


class HotelRoom(pydantic.BaseModel):
    checkin_date: datetime.date = pydantic.Field(
        ...,
        description="The date of the check-in for the room (YYYY-MM-DD format)",
    )
    checkout_date: datetime.date = pydantic.Field(
        ...,
        description="The date of the check-out for the room (YYYY-MM-DD format)",
    )
    room_type: str = pydantic.Field(
        ...,
        description="The type of the room",
        enum=[x.value for x in RoomType],
    )
    num_guests: int = pydantic.Field(
        ...,
        description="The number of guests in the room",
    )


class BookHotelToolParameters(pydantic.BaseModel):
    hotel_id: str = pydantic.Field(
        ...,
        description="The ID of the hotel to book a room at",
    )
    rooms: list[HotelRoom] = pydantic.Field(
        ...,
        description="The rooms to book with the corresponding parameters",
        example=[
            {
                "checkin_date": "2025-12-15",
                "checkout_date": "2025-12-17",
                "room_type": "double",
                "num_guests": 2,
            },
            {
                "checkin_date": "2025-12-15",
                "checkout_date": "2025-12-18",
                "room_type": "single",
                "num_guests": 1,
            }
        ],
    )


class BookHotelTool(BaseTool):
    def __init__(
        self,
        db: HotelDB,
        name: str = "book_hotel",
        description: str = "Book a hotel with the given parameters",
    ):
        super().__init__(name, description, BookHotelToolParameters)
        self.db = db

    def __call__(self, parameters: BookHotelToolParameters) -> dict:
        return {"success": True, "message": "Hotel room booked successfully"}


class SearchTrainsToolParameters(pydantic.BaseModel):
    departure: str = pydantic.Field(
        ...,
        description="The departure station",
        enum=[x.value for x in Station],
    )
    destination: str = pydantic.Field(
        ...,
        description="The destination station",
        enum=[x.value for x in Station],
    )
    weekday: str = pydantic.Field(
        None,
        description="The weekday of the journey",
        enum=[x.value for x in Weekday],
    )
    leave_before: datetime.time = pydantic.Field(
        None,
        description="Search for trains leaving departure station before this time",
    )
    leave_after: datetime.time = pydantic.Field(
        None,
        description="Search for trains leaving departure station after this time",
    )
    arrive_before: datetime.time = pydantic.Field(
        None,
        description="Search for trains arriving at destination station before this time",
    )
    arrive_after: datetime.time = pydantic.Field(
        None,
        description="Search for trains arriving at destination station after this time",
    )


class SearchTrainsTool(BaseSearchTool):
    def __init__(
        self,
        db: TrainDB,
        name: str = "search_trains",
        description: str = "Search for trains with the given parameters",
    ):
        super().__init__(db, name, description, SearchTrainsToolParameters)

    def __call__(self, parameters: SearchTrainsToolParameters) -> list[dict]:
        parameters.departure = parameters.departure.lower() if parameters.departure else None
        parameters.destination = parameters.destination.lower() if parameters.destination else None
        parameters.weekday = parameters.weekday.lower() if parameters.weekday else None

        params_dict = parameters.model_dump()
        for time_field in ['leave_after', 'leave_before', 'arrive_after', 'arrive_before']:
            if params_dict.get(time_field) is not None:
                params_dict[time_field] = params_dict[time_field].isoformat()

        result = self.db.query(**params_dict)
        return [r.to_dict() for r in result]


class TrainTicket(pydantic.BaseModel):
    ticket_type: str = pydantic.Field(
        "one-way",
        description="The type of the ticket",
        enum=["one-way", "return"],
    )
    discount: float = pydantic.Field(
        0.0,
        description="The discount to apply to the ticket",
    )
    first_class: bool = pydantic.Field(
        False,
        description="Whether the ticket is for first class",
    )


class BuyTrainTicketsToolParameters(pydantic.BaseModel):
    train_id: str = pydantic.Field(
        ...,
        description="The ID of the train to buy tickets for",
    )
    tickets: list[TrainTicket] = pydantic.Field(
        ...,
        description="The parameters for the tickets to buy",
        example=[
            {
                "ticket_type": "one-way",
                "discount": 0.5,
                "first_class": False,
            },
            {
                "ticket_type": "return",
                "discount": 0.0,
                "first_class": True,
            },
        ],
    )


class BuyTrainTicketsTool(BaseTool):
    def __init__(
        self,
        db: TrainDB,
        name: str = "buy_train_tickets",
        description: str = "Buy train tickets for the selected train",
    ):
        super().__init__(name, description, BuyTrainTicketsToolParameters)
        self.db = db

    def __call__(self, parameters: BuyTrainTicketsToolParameters) -> dict:
        return {"success": True, "message": "Train tickets purchased successfully"}
