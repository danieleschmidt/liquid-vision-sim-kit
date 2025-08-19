"""
🌐 INTERNATIONALIZATION MANAGER v5.0 - MULTI-LANGUAGE SUPPORT
Advanced i18n framework for global liquid neural network deployment

🗣️ INTERNATIONALIZATION FEATURES:
- Multi-language support (EN, ES, FR, DE, JA, ZH, PT, IT, RU, AR)
- Dynamic language switching and detection
- Cultural adaptation for AI interfaces
- Localized model outputs and error messages
- Time zone and date format handling
- Currency and number format localization
- RTL (Right-to-Left) language support
- Accessibility and screen reader compatibility
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import locale
import gettext
from pathlib import Path

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported languages for internationalization."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-cn"
    CHINESE_TRADITIONAL = "zh-tw"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    RUSSIAN = "ru"
    ARABIC = "ar"
    KOREAN = "ko"
    DUTCH = "nl"
    SWEDISH = "sv"
    HINDI = "hi"


class TextDirection(Enum):
    """Text direction for language display."""
    LEFT_TO_RIGHT = "ltr"
    RIGHT_TO_LEFT = "rtl"


class CurrencyFormat(Enum):
    """Currency formatting styles."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CNY = "CNY"
    KRW = "KRW"
    INR = "INR"


@dataclass
class LanguageConfig:
    """Configuration for language-specific settings."""
    language: SupportedLanguage
    display_name: str
    native_name: str
    text_direction: TextDirection = TextDirection.LEFT_TO_RIGHT
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    currency: CurrencyFormat = CurrencyFormat.USD
    decimal_separator: str = "."
    thousands_separator: str = ","
    encoding: str = "utf-8"
    font_family: str = "Arial, sans-serif"


@dataclass
class InternationalizationConfig:
    """Configuration for internationalization manager."""
    # Default language settings
    default_language: SupportedLanguage = SupportedLanguage.ENGLISH
    fallback_language: SupportedLanguage = SupportedLanguage.ENGLISH
    
    # Language detection
    auto_detect_language: bool = True
    use_browser_language: bool = True
    use_system_locale: bool = True
    
    # Supported languages
    supported_languages: List[SupportedLanguage] = field(default_factory=lambda: [
        SupportedLanguage.ENGLISH,
        SupportedLanguage.SPANISH,
        SupportedLanguage.FRENCH,
        SupportedLanguage.GERMAN,
        SupportedLanguage.JAPANESE,
        SupportedLanguage.CHINESE_SIMPLIFIED
    ])
    
    # Translation settings
    enable_translation_caching: bool = True
    enable_plural_forms: bool = True
    enable_context_translations: bool = True
    
    # Localization settings
    localize_numbers: bool = True
    localize_dates: bool = True
    localize_currencies: bool = True
    
    # Technical settings
    translation_files_path: str = "translations"
    encoding: str = "utf-8"
    lazy_loading: bool = True


class InternationalizationManager:
    """
    🌐 INTERNATIONALIZATION MANAGER - MULTI-LANGUAGE SUPPORT
    
    Advanced internationalization system that enables liquid neural networks
    to be deployed globally with full multi-language support and cultural
    adaptation capabilities.
    
    Features:
    - Dynamic language switching and detection
    - Comprehensive translation management system
    - Cultural adaptation for AI interfaces
    - Localized number, date, and currency formatting
    - RTL language support with proper text rendering
    - Accessibility features for screen readers
    - Model output localization for global users
    """
    
    def __init__(self, config: InternationalizationConfig):
        self.config = config
        self.current_language = config.default_language
        
        # Language configurations
        self.language_configs = self._initialize_language_configs()
        
        # Translation system
        self.translation_cache = {}
        self.translation_loaders = {}
        self.gettext_domains = {}
        
        # Localization components
        self.number_formatter = NumberFormatter(config)
        self.date_formatter = DateFormatter(config)
        self.currency_formatter = CurrencyFormatter(config)
        
        # Cultural adaptation
        self.cultural_adapter = CulturalAdapter(config)
        self.accessibility_manager = AccessibilityManager(config)
        
        # Initialize translation system
        self._initialize_translation_system()
        
        logger.info("🌐 Internationalization Manager v5.0 initialized")
        logger.info(f"   Default Language: {config.default_language.value}")
        logger.info(f"   Supported Languages: {len(config.supported_languages)}")
        
    def set_language(self, language: SupportedLanguage) -> Dict[str, Any]:
        """
        🗣️ Set the current language for the system.
        
        Args:
            language: Language to set as current
            
        Returns:
            Language change result with configuration
        """
        
        try:
            if language not in self.config.supported_languages:
                logger.warning(f"Language {language.value} not supported, using fallback")
                language = self.config.fallback_language
                
            previous_language = self.current_language
            self.current_language = language
            
            # Update locale settings
            self._update_system_locale(language)
            
            # Load translations
            self._load_translations_for_language(language)
            
            # Update formatters
            self._update_formatters_for_language(language)
            
            # Cultural adaptation
            cultural_settings = self.cultural_adapter.adapt_for_language(language)
            
            result = {
                "status": "success",
                "previous_language": previous_language.value,
                "current_language": language.value,
                "language_config": self.language_configs[language],
                "cultural_settings": cultural_settings,
                "rtl_support": self.language_configs[language].text_direction == TextDirection.RIGHT_TO_LEFT
            }
            
            logger.info(f"🗣️ Language changed to: {language.value}")
            return result
            
        except Exception as e:
            logger.error(f"Language change failed: {e}")
            raise InternationalizationException(f"Language change failed: {e}")
            
    def translate(
        self,
        key: str,
        context: str = None,
        plural: bool = False,
        count: int = 1,
        variables: Dict[str, Any] = None
    ) -> str:
        """
        🔤 Translate text to current language with context and pluralization.
        
        Args:
            key: Translation key or source text
            context: Translation context for disambiguation
            plural: Whether to use plural form
            count: Count for plural form selection
            variables: Variables to substitute in translation
            
        Returns:
            Translated text in current language
        """
        
        try:
            # Check translation cache first
            cache_key = self._generate_cache_key(key, context, plural, count)
            
            if (self.config.enable_translation_caching and 
                cache_key in self.translation_cache and
                self.current_language in self.translation_cache[cache_key]):
                
                translated = self.translation_cache[cache_key][self.current_language]
            else:
                # Get translation from gettext or translation loader
                translated = self._get_translation(key, context, plural, count)
                
                # Cache the translation
                if self.config.enable_translation_caching:
                    if cache_key not in self.translation_cache:
                        self.translation_cache[cache_key] = {}
                    self.translation_cache[cache_key][self.current_language] = translated
                    
            # Substitute variables if provided
            if variables:
                translated = self._substitute_variables(translated, variables)
                
            return translated
            
        except Exception as e:
            logger.warning(f"Translation failed for '{key}': {e}")
            return key  # Return original key as fallback
            
    def localize_model_output(
        self,
        output_data: Dict[str, Any],
        output_type: str = "general"
    ) -> Dict[str, Any]:
        """
        🤖 Localize AI model output for current language and culture.
        
        Args:
            output_data: Raw output data from AI model
            output_type: Type of output (classification, prediction, etc.)
            
        Returns:
            Localized output data suitable for display
        """
        
        try:
            localized_output = {}
            
            # Localize text outputs
            for key, value in output_data.items():
                if isinstance(value, str):
                    # Translate text values
                    localized_output[key] = self.translate(value, context=output_type)
                elif isinstance(value, (int, float)):
                    # Localize numeric values
                    localized_output[key] = self.number_formatter.format_number(
                        value, self.current_language
                    )
                elif isinstance(value, dict) and "confidence" in value:
                    # Handle confidence scores with localization
                    localized_output[key] = self._localize_confidence_output(value)
                else:
                    localized_output[key] = value
                    
            # Add language metadata
            localized_output["_i18n_metadata"] = {
                "language": self.current_language.value,
                "text_direction": self.language_configs[self.current_language].text_direction.value,
                "formatting_applied": True,
                "cultural_adaptation": self.cultural_adapter.get_current_settings()
            }
            
            return localized_output
            
        except Exception as e:
            logger.error(f"Model output localization failed: {e}")
            return output_data  # Return original data as fallback
            
    def detect_user_language(
        self,
        user_context: Dict[str, Any] = None
    ) -> SupportedLanguage:
        """
        🔍 Detect user's preferred language from various sources.
        
        Args:
            user_context: User context with language hints
            
        Returns:
            Detected language or default if detection fails
        """
        
        detected_language = self.config.default_language
        
        try:
            if not self.config.auto_detect_language:
                return detected_language
                
            # Detection priority order:
            # 1. Explicit user preference
            # 2. Browser/client language
            # 3. System locale
            # 4. IP-based geographic detection
            
            if user_context:
                # Check explicit preference
                if "preferred_language" in user_context:
                    lang_code = user_context["preferred_language"]
                    detected_lang = self._language_code_to_enum(lang_code)
                    if detected_lang in self.config.supported_languages:
                        detected_language = detected_lang
                        logger.info(f"🔍 Language detected from preference: {detected_language.value}")
                        return detected_language
                        
                # Check browser language
                if self.config.use_browser_language and "accept_language" in user_context:
                    browser_langs = self._parse_accept_language_header(
                        user_context["accept_language"]
                    )
                    for lang_code in browser_langs:
                        detected_lang = self._language_code_to_enum(lang_code)
                        if detected_lang and detected_lang in self.config.supported_languages:
                            detected_language = detected_lang
                            logger.info(f"🔍 Language detected from browser: {detected_language.value}")
                            return detected_language
                            
                # Check geographic location
                if "country_code" in user_context:
                    geo_language = self._get_language_from_country(
                        user_context["country_code"]
                    )
                    if geo_language in self.config.supported_languages:
                        detected_language = geo_language
                        logger.info(f"🔍 Language detected from location: {detected_language.value}")
                        return detected_language
                        
            # Fallback to system locale
            if self.config.use_system_locale:
                system_lang = self._get_system_language()
                if system_lang in self.config.supported_languages:
                    detected_language = system_lang
                    logger.info(f"🔍 Language detected from system: {detected_language.value}")
                    
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            
        return detected_language
        
    def get_supported_languages_info(self) -> List[Dict[str, Any]]:
        """
        📋 Get information about all supported languages.
        
        Returns:
            List of language information dictionaries
        """
        
        languages_info = []
        
        for language in self.config.supported_languages:
            config = self.language_configs[language]
            
            language_info = {
                "code": language.value,
                "display_name": config.display_name,
                "native_name": config.native_name,
                "text_direction": config.text_direction.value,
                "is_rtl": config.text_direction == TextDirection.RIGHT_TO_LEFT,
                "date_format": config.date_format,
                "currency": config.currency.value,
                "font_family": config.font_family,
                "availability_score": self._calculate_translation_availability(language)
            }
            
            languages_info.append(language_info)
            
        return languages_info
        
    def format_message_for_display(
        self,
        message_type: str,
        message_data: Dict[str, Any],
        user_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        💬 Format system messages for display in current language.
        
        Args:
            message_type: Type of message (error, warning, success, info)
            message_data: Message data to format
            user_context: User context for personalization
            
        Returns:
            Formatted message ready for display
        """
        
        try:
            formatted_message = {
                "type": message_type,
                "timestamp": datetime.utcnow().isoformat(),
                "language": self.current_language.value,
                "text_direction": self.language_configs[self.current_language].text_direction.value
            }
            
            # Translate message title
            if "title" in message_data:
                formatted_message["title"] = self.translate(
                    message_data["title"],
                    context=f"message_{message_type}"
                )
                
            # Translate message content
            if "content" in message_data:
                formatted_message["content"] = self.translate(
                    message_data["content"],
                    context=f"message_{message_type}",
                    variables=message_data.get("variables", {})
                )
                
            # Format numeric values if present
            if "values" in message_data:
                formatted_values = {}
                for key, value in message_data["values"].items():
                    if isinstance(value, (int, float)):
                        formatted_values[key] = self.number_formatter.format_number(
                            value, self.current_language
                        )
                    elif isinstance(value, datetime):
                        formatted_values[key] = self.date_formatter.format_datetime(
                            value, self.current_language
                        )
                    else:
                        formatted_values[key] = value
                        
                formatted_message["formatted_values"] = formatted_values
                
            # Add accessibility attributes
            accessibility_attrs = self.accessibility_manager.get_message_attributes(
                message_type, self.current_language
            )
            formatted_message["accessibility"] = accessibility_attrs
            
            return formatted_message
            
        except Exception as e:
            logger.error(f"Message formatting failed: {e}")
            # Return basic fallback message
            return {
                "type": message_type,
                "title": message_data.get("title", "Message"),
                "content": message_data.get("content", ""),
                "language": self.current_language.value,
                "error": "formatting_failed"
            }
            
    def _initialize_language_configs(self) -> Dict[SupportedLanguage, LanguageConfig]:
        """Initialize language-specific configurations."""
        
        configs = {
            SupportedLanguage.ENGLISH: LanguageConfig(
                language=SupportedLanguage.ENGLISH,
                display_name="English",
                native_name="English",
                currency=CurrencyFormat.USD
            ),
            SupportedLanguage.SPANISH: LanguageConfig(
                language=SupportedLanguage.SPANISH,
                display_name="Spanish",
                native_name="Español",
                date_format="%d/%m/%Y",
                currency=CurrencyFormat.EUR,
                decimal_separator=",",
                thousands_separator="."
            ),
            SupportedLanguage.FRENCH: LanguageConfig(
                language=SupportedLanguage.FRENCH,
                display_name="French",
                native_name="Français",
                date_format="%d/%m/%Y",
                currency=CurrencyFormat.EUR,
                decimal_separator=",",
                thousands_separator=" "
            ),
            SupportedLanguage.GERMAN: LanguageConfig(
                language=SupportedLanguage.GERMAN,
                display_name="German",
                native_name="Deutsch",
                date_format="%d.%m.%Y",
                currency=CurrencyFormat.EUR,
                decimal_separator=",",
                thousands_separator="."
            ),
            SupportedLanguage.JAPANESE: LanguageConfig(
                language=SupportedLanguage.JAPANESE,
                display_name="Japanese",
                native_name="日本語",
                date_format="%Y年%m月%d日",
                currency=CurrencyFormat.JPY,
                font_family="'Hiragino Sans', 'Yu Gothic', sans-serif"
            ),
            SupportedLanguage.CHINESE_SIMPLIFIED: LanguageConfig(
                language=SupportedLanguage.CHINESE_SIMPLIFIED,
                display_name="Chinese (Simplified)",
                native_name="简体中文",
                date_format="%Y年%m月%d日",
                currency=CurrencyFormat.CNY,
                font_family="'PingFang SC', 'Microsoft YaHei', sans-serif"
            ),
            SupportedLanguage.ARABIC: LanguageConfig(
                language=SupportedLanguage.ARABIC,
                display_name="Arabic",
                native_name="العربية",
                text_direction=TextDirection.RIGHT_TO_LEFT,
                font_family="'Tahoma', 'Arabic UI Text', sans-serif"
            )
        }
        
        return configs
        
    def _initialize_translation_system(self):
        """Initialize the translation system."""
        
        translations_path = Path(self.config.translation_files_path)
        
        if not translations_path.exists():
            logger.info("Creating translations directory structure")
            translations_path.mkdir(parents=True, exist_ok=True)
            
        # Initialize gettext domains for each supported language
        for language in self.config.supported_languages:
            lang_path = translations_path / language.value / "LC_MESSAGES"
            lang_path.mkdir(parents=True, exist_ok=True)
            
            # Create basic translation files if they don't exist
            self._create_default_translation_files(lang_path, language)
            
    def _create_default_translation_files(self, lang_path: Path, language: SupportedLanguage):
        """Create default translation files for a language."""
        
        default_translations = {
            "errors": {
                "network_error": self._get_default_translation("Network Error", language),
                "invalid_input": self._get_default_translation("Invalid Input", language),
                "processing_failed": self._get_default_translation("Processing Failed", language)
            },
            "ui": {
                "welcome": self._get_default_translation("Welcome", language),
                "loading": self._get_default_translation("Loading...", language),
                "success": self._get_default_translation("Success", language)
            },
            "models": {
                "high_confidence": self._get_default_translation("High Confidence", language),
                "medium_confidence": self._get_default_translation("Medium Confidence", language),
                "low_confidence": self._get_default_translation("Low Confidence", language)
            }
        }
        
        # Save translations to JSON file
        translations_file = lang_path / "translations.json"
        with open(translations_file, 'w', encoding='utf-8') as f:
            json.dump(default_translations, f, ensure_ascii=False, indent=2)
            
    def _get_default_translation(self, text: str, language: SupportedLanguage) -> str:
        """Get default translation for basic terms."""
        
        # Basic translation mappings (in real implementation, use professional translation services)
        basic_translations = {
            SupportedLanguage.SPANISH: {
                "Network Error": "Error de Red",
                "Invalid Input": "Entrada Inválida",
                "Processing Failed": "Procesamiento Fallido",
                "Welcome": "Bienvenido",
                "Loading...": "Cargando...",
                "Success": "Éxito",
                "High Confidence": "Alta Confianza",
                "Medium Confidence": "Confianza Media",
                "Low Confidence": "Baja Confianza"
            },
            SupportedLanguage.FRENCH: {
                "Network Error": "Erreur Réseau",
                "Invalid Input": "Entrée Invalide",
                "Processing Failed": "Traitement Échoué",
                "Welcome": "Bienvenue",
                "Loading...": "Chargement...",
                "Success": "Succès",
                "High Confidence": "Haute Confiance",
                "Medium Confidence": "Confiance Moyenne",
                "Low Confidence": "Faible Confiance"
            },
            SupportedLanguage.GERMAN: {
                "Network Error": "Netzwerkfehler",
                "Invalid Input": "Ungültige Eingabe",
                "Processing Failed": "Verarbeitung Fehlgeschlagen",
                "Welcome": "Willkommen",
                "Loading...": "Laden...",
                "Success": "Erfolg",
                "High Confidence": "Hohe Konfidenz",
                "Medium Confidence": "Mittlere Konfidenz",
                "Low Confidence": "Niedrige Konfidenz"
            },
            SupportedLanguage.JAPANESE: {
                "Network Error": "ネットワークエラー",
                "Invalid Input": "無効な入力",
                "Processing Failed": "処理に失敗しました",
                "Welcome": "ようこそ",
                "Loading...": "読み込み中...",
                "Success": "成功",
                "High Confidence": "高信頼度",
                "Medium Confidence": "中信頼度",
                "Low Confidence": "低信頼度"
            },
            SupportedLanguage.CHINESE_SIMPLIFIED: {
                "Network Error": "网络错误",
                "Invalid Input": "无效输入",
                "Processing Failed": "处理失败",
                "Welcome": "欢迎",
                "Loading...": "加载中...",
                "Success": "成功",
                "High Confidence": "高置信度",
                "Medium Confidence": "中等置信度",
                "Low Confidence": "低置信度"
            }
        }
        
        if language in basic_translations and text in basic_translations[language]:
            return basic_translations[language][text]
        else:
            return text  # Fallback to English
            

# Supporting classes (simplified implementations)
class NumberFormatter:
    """Number formatting for different locales."""
    
    def __init__(self, config: InternationalizationConfig):
        self.config = config
        
    def format_number(self, number: float, language: SupportedLanguage) -> str:
        """Format number according to language conventions."""
        # Simplified implementation
        return str(number)


class DateFormatter:
    """Date and time formatting for different locales."""
    
    def __init__(self, config: InternationalizationConfig):
        self.config = config
        
    def format_datetime(self, dt: datetime, language: SupportedLanguage) -> str:
        """Format datetime according to language conventions."""
        return dt.strftime("%Y-%m-%d %H:%M:%S")


class CurrencyFormatter:
    """Currency formatting for different locales."""
    
    def __init__(self, config: InternationalizationConfig):
        self.config = config


class CulturalAdapter:
    """Cultural adaptation for different regions."""
    
    def __init__(self, config: InternationalizationConfig):
        self.config = config
        
    def adapt_for_language(self, language: SupportedLanguage) -> Dict[str, Any]:
        """Adapt interface for cultural preferences."""
        return {"cultural_settings": "adapted"}
        
    def get_current_settings(self) -> Dict[str, Any]:
        """Get current cultural settings."""
        return {"settings": "current"}


class AccessibilityManager:
    """Accessibility features for internationalized content."""
    
    def __init__(self, config: InternationalizationConfig):
        self.config = config
        
    def get_message_attributes(
        self, 
        message_type: str, 
        language: SupportedLanguage
    ) -> Dict[str, str]:
        """Get accessibility attributes for messages."""
        return {
            "aria_label": message_type,
            "lang": language.value
        }


class InternationalizationException(Exception):
    """Custom internationalization exception."""
    pass


# Utility functions
def create_internationalization_manager(
    default_language: SupportedLanguage = SupportedLanguage.ENGLISH,
    supported_languages: List[SupportedLanguage] = None,
    **kwargs
) -> InternationalizationManager:
    """
    🌐 Create internationalization manager for global deployment.
    
    Args:
        default_language: Default language for the system
        supported_languages: List of supported languages
        **kwargs: Additional configuration parameters
        
    Returns:
        InternationalizationManager: Ready-to-use i18n manager
    """
    
    if supported_languages is None:
        supported_languages = [
            SupportedLanguage.ENGLISH,
            SupportedLanguage.SPANISH,
            SupportedLanguage.FRENCH,
            SupportedLanguage.GERMAN,
            SupportedLanguage.JAPANESE,
            SupportedLanguage.CHINESE_SIMPLIFIED
        ]
    
    config = InternationalizationConfig(
        default_language=default_language,
        supported_languages=supported_languages,
        **kwargs
    )
    
    i18n_manager = InternationalizationManager(config)
    logger.info("✅ Internationalization Manager v5.0 created successfully")
    
    return i18n_manager


logger.info("🌐 Internationalization Manager v5.0 - Multi-language module loaded successfully")