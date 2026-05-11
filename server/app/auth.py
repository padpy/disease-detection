from functools import wraps
from flask import request, jsonify
from flask import Flask
from firebase_admin import auth

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        bearer = request.headers.get('Authorization').split(' ')[1]

        try:
            decoded_token = auth.verify_id_token(bearer, check_revoked=True)
            print(decoded_token['email'])
            return(f(*args, **kwargs))
        except auth.RevokedIdTokenError:
            return jsonify({"error": "Invalid authorization token"}), 401
        except auth.ExpiredIdTokenError:
            return jsonify({"error": "Authorization token has expired"}), 401
        except auth.InvalidIdTokenError:
            return jsonify({"error": "Invalid authorization token"}), 401
        except Exception as e:
            return jsonify({"error": "Unknown error"}), 401
        
    return decorated_function
