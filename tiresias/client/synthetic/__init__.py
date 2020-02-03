import os
import barnum
import pandas as pd
from time import time
from random import randint, random, choice
from tiresias.client.storage import register_app, insert_payload

def sample_profile():
    payload = {}
    zipcode, city, state = barnum.create_city_state_zip()
    payload["demographics"] = [{
        "age": randint(18, 100),
        "gender": choice(["Male", "Female", "Other"]),
        "income": randint(10, 1000) * 1000,
        "city": city,
        "state": state,
        "zipcode": zipcode
    }]
    return payload

def sample_browsing(profile):
    age = profile["demographics"][0]["age"] / 100.0
    payload = {}
    payload["history"] = []
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "domains.csv"))
    for _ in range(100 - randint(0, int(99 * age))):
        payload["history"].append({
            "timestamp": time() + randint(0, 10000),
            "domain": choice(df["Root Domain"].values)
        })
    return payload

def sample_screen_time(profile):
    payload = {}
    payload["types"] = [
        {"application_name": "Chrome", "application_type": "browser"},
        {"application_name": "Safari", "application_type": "browser"},
        {"application_name": "Firefox", "application_type": "browser"},
        {"application_name": "Microsoft Edge", "application_type": "browser"},
        {"application_name": "Internet Explorer", "application_type": "browser"},
        {"application_name": "VSCode", "application_type": "development"},
        {"application_name": "Terminal", "application_type": "development"},
        {"application_name": "iTerm", "application_type": "development"},
        {"application_name": "Slack", "application_type": "communication"},
        {"application_name": "Skype", "application_type": "communication"},
        {"application_name": "Zoom", "application_type": "communication"},
    ]
    payload["events"] = []
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "domains.csv"))
    for _ in range(randint(1, 100)):
        timestamp = time() + randint(0, 10000)
        application_name = choice(payload["types"])["application_name"]
        payload["events"].append({
            "timestamp": timestamp,
            "event_type": "open",
            "application_name": application_name
        })
        payload["events"].append({
            "timestamp": timestamp + randint(0, 10000),
            "event_type": "close",
            "application_name": application_name
        })
    return payload

def create_synthetic_dataset(storage_dir):
    register_app(storage_dir, "profile", {
        "demographics": {
            "description": "",
            "columns": {
                "age": {"type": "float", "description": ""},
                "gender": {"type": "float", "description": ""},
                "income": {"type": "float", "description": ""},
                "city": {"type": "float", "description": ""},
                "state": {"type": "float", "description": ""},
                "zipcode": {"type": "float", "description": ""},
            }
        }
    })
    profile = sample_profile()
    insert_payload(storage_dir, "profile", profile)

    register_app(storage_dir, "browsing", {
        "history": {
            "description": "",
            "columns": {
                "timestamp": {"type": "float", "description": "When the website was opened"},
                "domain": {"type": "float", "description": "The domain (i.e. everything up to `.com`, `.net`, etc.)"},
            }
        },
    })
    insert_payload(storage_dir, "browsing", sample_browsing(profile))

    register_app(storage_dir, "screen_time", {
        "events": {
            "description": "",
            "columns": {
                "timestamp": {"type": "float", "description": "When the event occurred."},
                "event_type": {"type": "string", "description": "Whether the application was opened or closed."},
                "application_name": {"type": "string", "description": "Standardized name for the application."},
            }
        },
        "types": {
            "description": "",
            "columns": {
                "application_name": {"type": "string", "description": "Standardized name for the application."},
                "application_type": {"type": "string", "description": "Whether the application is for web browsing, software development, or communicaation."},
            }
        },
    })
    insert_payload(storage_dir, "screen_time", sample_screen_time(profile))
