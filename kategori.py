def kategori_ulasan(text):

    driver_keywords = [
        "driver",
        "abang",
        "kurir",
        "ojek"
    ]

    sistem_keywords = [
        "aplikasi",
        "app",
        "sistem",
        "error",
        "bug",
        "login"
    ]

    text = text.lower()

    for k in driver_keywords:
        if k in text:
            return "Driver"

    for k in sistem_keywords:
        if k in text:
            return "Sistem"

    return "Lainnya"