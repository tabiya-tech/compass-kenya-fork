import { getDefaultLocale } from "../envService";

// Default locale, now loaded from environment variable FRONTEND_DEFAULT_LOCALE (base64-encoded in env.js)
export const DEFAULT_LOCALE = getDefaultLocale();

export enum Locale {
  EN_GB = "en-GB",
  EN_US = "en-US",
  ES_ES = "es-ES",
  ES_AR = "es-AR",
  SW_KE = "sw-KE",
}

export const LocalesLabels = {
  [Locale.EN_GB]: "English (UK)",
  [Locale.EN_US]: "English (US)",
  [Locale.ES_ES]: "Español (España)",
  [Locale.ES_AR]: "Español (Argentina)",
  [Locale.SW_KE]: "Kiswahili (Kenya)",
} as const;

export const SupportedLocales: Locale[] = [Locale.EN_GB, Locale.EN_US, Locale.ES_ES, Locale.ES_AR, Locale.SW_KE];
export const FALL_BACK_LOCALE = Locale.EN_GB;
