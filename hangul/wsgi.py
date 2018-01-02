"""
WSGI config for hangul project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/1.11/howto/deployment/wsgi/
"""

import os
import sys
from django.core.wsgi import get_wsgi_application

sys.path.append('/home/waylight3/hangul')
sys.path.append('/home/waylight3/hangul/hangul')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "hangul.settings")

application = get_wsgi_application()
