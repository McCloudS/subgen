from enum import Enum

class LanguageCode(Enum):
    # ISO 639-1, ISO 639-2/T, ISO 639-2/B, English Name, Native Name
    AFAR = ("aa", "aar", "aar", "Afar", "Afar") 
    AFRIKAANS = ("af", "afr", "afr", "Afrikaans", "Afrikaans")
    AMHARIC = ("am", "amh", "amh", "Amharic", "አማርኛ")
    ARABIC = ("ar", "ara", "ara", "Arabic", "العربية")
    ASSAMESE = ("as", "asm", "asm", "Assamese", "অসমীয়া")
    AZERBAIJANI = ("az", "aze", "aze", "Azerbaijani", "Azərbaycanca")
    BASHKIR = ("ba", "bak", "bak", "Bashkir", "Башҡортса")
    BELARUSIAN = ("be", "bel", "bel", "Belarusian", "Беларуская")
    BULGARIAN = ("bg", "bul", "bul", "Bulgarian", "Български")
    BENGALI = ("bn", "ben", "ben", "Bengali", "বাংলা")
    TIBETAN = ("bo", "bod", "tib", "Tibetan", "བོད་ཡིག")
    BRETON = ("br", "bre", "bre", "Breton", "Brezhoneg")
    BOSNIAN = ("bs", "bos", "bos", "Bosnian", "Bosanski")
    CATALAN = ("ca", "cat", "cat", "Catalan", "Català")
    CZECH = ("cs", "ces", "cze", "Czech", "Čeština")
    WELSH = ("cy", "cym", "wel", "Welsh", "Cymraeg")
    DANISH = ("da", "dan", "dan", "Danish", "Dansk")
    GERMAN = ("de", "deu", "ger", "German", "Deutsch")
    GREEK = ("el", "ell", "gre", "Greek", "Ελληνικά")
    ENGLISH = ("en", "eng", "eng", "English", "English")
    SPANISH = ("es", "spa", "spa", "Spanish", "Español")
    ESTONIAN = ("et", "est", "est", "Estonian", "Eesti")
    BASQUE = ("eu", "eus", "baq", "Basque", "Euskara")
    PERSIAN = ("fa", "fas", "per", "Persian", "فارسی")
    FINNISH = ("fi", "fin", "fin", "Finnish", "Suomi")
    FAROESE = ("fo", "fao", "fao", "Faroese", "Føroyskt")
    FRENCH = ("fr", "fra", "fre", "French", "Français")
    GALICIAN = ("gl", "glg", "glg", "Galician", "Galego")
    GUJARATI = ("gu", "guj", "guj", "Gujarati", "ગુજરાતી")
    HAUSA = ("ha", "hau", "hau", "Hausa", "Hausa")
    HAWAIIAN = ("haw", "haw", "haw", "Hawaiian", "ʻŌlelo Hawaiʻi")
    HEBREW = ("he", "heb", "heb", "Hebrew", "עברית")
    HINDI = ("hi", "hin", "hin", "Hindi", "हिन्दी")
    CROATIAN = ("hr", "hrv", "hrv", "Croatian", "Hrvatski")
    HAITIAN_CREOLE = ("ht", "hat", "hat", "Haitian Creole", "Kreyòl Ayisyen")
    HUNGARIAN = ("hu", "hun", "hun", "Hungarian", "Magyar")
    ARMENIAN = ("hy", "hye", "arm", "Armenian", "Հայերեն")
    INDONESIAN = ("id", "ind", "ind", "Indonesian", "Bahasa Indonesia")
    ICELANDIC = ("is", "isl", "ice", "Icelandic", "Íslenska")
    ITALIAN = ("it", "ita", "ita", "Italian", "Italiano")
    JAPANESE = ("ja", "jpn", "jpn", "Japanese", "日本語")
    JAVANESE = ("jw", "jav", "jav", "Javanese", "ꦧꦱꦗꦮ")
    GEORGIAN = ("ka", "kat", "geo", "Georgian", "ქართული")
    KAZAKH = ("kk", "kaz", "kaz", "Kazakh", "Қазақша")
    KHMER = ("km", "khm", "khm", "Khmer", "ភាសាខ្មែរ")
    KANNADA = ("kn", "kan", "kan", "Kannada", "ಕನ್ನಡ")
    KOREAN = ("ko", "kor", "kor", "Korean", "한국어")
    LATIN = ("la", "lat", "lat", "Latin", "Latina")
    LUXEMBOURGISH = ("lb", "ltz", "ltz", "Luxembourgish", "Lëtzebuergesch")
    LINGALA = ("ln", "lin", "lin", "Lingala", "Lingála")
    LAO = ("lo", "lao", "lao", "Lao", "ພາສາລາວ")
    LITHUANIAN = ("lt", "lit", "lit", "Lithuanian", "Lietuvių")
    LATVIAN = ("lv", "lav", "lav", "Latvian", "Latviešu")
    MALAGASY = ("mg", "mlg", "mlg", "Malagasy", "Malagasy")
    MAORI = ("mi", "mri", "mao", "Maori", "Te Reo Māori")
    MACEDONIAN = ("mk", "mkd", "mac", "Macedonian", "Македонски")
    MALAYALAM = ("ml", "mal", "mal", "Malayalam", "മലയാളം")
    MONGOLIAN = ("mn", "mon", "mon", "Mongolian", "Монгол")
    MARATHI = ("mr", "mar", "mar", "Marathi", "मराठी")
    MALAY = ("ms", "msa", "may", "Malay", "Bahasa Melayu")
    MALTESE = ("mt", "mlt", "mlt", "Maltese", "Malti")
    BURMESE = ("my", "mya", "bur", "Burmese", "မြန်မာစာ")
    NEPALI = ("ne", "nep", "nep", "Nepali", "नेपाली")
    DUTCH = ("nl", "nld", "dut", "Dutch", "Nederlands")
    NORWEGIAN_NYNORSK = ("nn", "nno", "nno", "Norwegian Nynorsk", "Nynorsk")
    NORWEGIAN = ("no", "nor", "nor", "Norwegian", "Norsk")
    OCCITAN = ("oc", "oci", "oci", "Occitan", "Occitan")
    PUNJABI = ("pa", "pan", "pan", "Punjabi", "ਪੰਜਾਬੀ")
    POLISH = ("pl", "pol", "pol", "Polish", "Polski")
    PASHTO = ("ps", "pus", "pus", "Pashto", "پښتو")
    PORTUGUESE = ("pt", "por", "por", "Portuguese", "Português")
    ROMANIAN = ("ro", "ron", "rum", "Romanian", "Română")
    RUSSIAN = ("ru", "rus", "rus", "Russian", "Русский")
    SANSKRIT = ("sa", "san", "san", "Sanskrit", "संस्कृतम्")
    SINDHI = ("sd", "snd", "snd", "Sindhi", "سنڌي")
    SINHALA = ("si", "sin", "sin", "Sinhala", "සිංහල")
    SLOVAK = ("sk", "slk", "slo", "Slovak", "Slovenčina")
    SLOVENE = ("sl", "slv", "slv", "Slovene", "Slovenščina")
    SHONA = ("sn", "sna", "sna", "Shona", "ChiShona")
    SOMALI = ("so", "som", "som", "Somali", "Soomaaliga")
    ALBANIAN = ("sq", "sqi", "alb", "Albanian", "Shqip")
    SERBIAN = ("sr", "srp", "srp", "Serbian", "Српски")
    SUNDANESE = ("su", "sun", "sun", "Sundanese", "Basa Sunda")
    SWEDISH = ("sv", "swe", "swe", "Swedish", "Svenska")
    SWAHILI = ("sw", "swa", "swa", "Swahili", "Kiswahili")
    TAMIL = ("ta", "tam", "tam", "Tamil", "தமிழ்")
    TELUGU = ("te", "tel", "tel", "Telugu", "తెలుగు")
    TAJIK = ("tg", "tgk", "tgk", "Tajik", "Тоҷикӣ")
    THAI = ("th", "tha", "tha", "Thai", "ไทย")
    TURKMEN = ("tk", "tuk", "tuk", "Turkmen", "Türkmençe")
    TAGALOG = ("tl", "tgl", "tgl", "Tagalog", "Tagalog")
    TURKISH = ("tr", "tur", "tur", "Turkish", "Türkçe")
    TATAR = ("tt", "tat", "tat", "Tatar", "Татарча")
    UKRAINIAN = ("uk", "ukr", "ukr", "Ukrainian", "Українська")
    URDU = ("ur", "urd", "urd", "Urdu", "اردو")
    UZBEK = ("uz", "uzb", "uzb", "Uzbek", "Oʻzbek")
    VIETNAMESE = ("vi", "vie", "vie", "Vietnamese", "Tiếng Việt")
    YIDDISH = ("yi", "yid", "yid", "Yiddish", "ייִדיש")
    YORUBA = ("yo", "yor", "yor", "Yoruba", "Yorùbá")
    CHINESE = ("zh", "zho", "chi", "Chinese", "中文")
    CANTONESE = ("yue", "yue", "yue", "Cantonese", "粵語")
    NONE = (None, None, None, None, None)  # For no language
    # und for Undetermined aka unknown language https://www.loc.gov/standards/iso639-2/faq.html#25

    def __init__(self, iso_639_1, iso_639_2_t, iso_639_2_b, name_en, name_native):
        self.iso_639_1 = iso_639_1
        self.iso_639_2_t = iso_639_2_t
        self.iso_639_2_b = iso_639_2_b
        self.name_en = name_en
        self.name_native = name_native

    @staticmethod
    def from_iso_639_1(code):
        for lang in LanguageCode:
            if lang.iso_639_1 == code:
                return lang
        return LanguageCode.NONE

    @staticmethod
    def from_iso_639_2(code):
        for lang in LanguageCode:
            if lang.iso_639_2_t == code or lang.iso_639_2_b == code:
                return lang
        return LanguageCode.NONE

    @staticmethod
    def from_name(name : str):
        """Convert a language name (either English or native) to LanguageCode enum."""
        for lang in LanguageCode:
            if lang.name_en.lower() == name.lower() or lang.name_native.lower() == name.lower():
                return lang
        LanguageCode.NONE
        

    @staticmethod    
    def from_string(value: str):
        """
        Convert a string to a LanguageCode instance. Matches on ISO codes, English name, or native name.
        """
        if value is None:
            return LanguageCode.NONE
        value = value.strip().lower()
        for lang in LanguageCode:
            if lang is LanguageCode.NONE:
                continue
            elif (
                value == lang.iso_639_1
                or value == lang.iso_639_2_t
                or value == lang.iso_639_2_b
                or value == lang.name_en.lower()
                or value == lang.name_native.lower()
            ):
                return lang
        return LanguageCode.NONE
    
    # is valid language
    @staticmethod
    def is_valid_language(language: str):
        return LanguageCode.from_string(language) is not LanguageCode.NONE
    
    def to_iso_639_1(self):
        return self.iso_639_1

    def to_iso_639_2_t(self):
        return self.iso_639_2_t

    def to_iso_639_2_b(self):
        return self.iso_639_2_b

    def to_name(self, in_english=True):
        return self.name_en if in_english else self.name_native
    def __str__(self):
        if self.name_en is None:
            return "Unknown"
        return self.name_en
    
    def __bool__(self):
        return True if self.iso_639_1 is not None else False
    
    def __eq__(self, other):
        """
        Compare the LanguageCode instance to another object.
        Explicitly handle comparison to None.
        """
        if other is None:
            # If compared to None, return False unless self is None
            return self.iso_639_1 is None
        if isinstance(other, str):  # Allow comparison with a string
            return self.value == LanguageCode.from_string(other)
        if isinstance(other, LanguageCode):
            # Normal comparison for LanguageCode instances
            return self.iso_639_1 == other.iso_639_1
        # Otherwise, defer to the default equality
        return NotImplemented
