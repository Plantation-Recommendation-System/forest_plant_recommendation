import axios from 'axios';

const API_URL =
  import.meta.env.VITE_API_URL ||
  'http://127.0.0.1:8000/api/v1/recommendations';


export async function fetchRecommendations({
  location = '',
  latitude = null,
  longitude = null,
  soilImage,
  wetBehavior = '',
  topN = 10,
}) {
  try {
    const hasGps =
      latitude !== null &&
      latitude !== undefined &&
      longitude !== null &&
      longitude !== undefined;

    const hasManualLocation =
      Boolean(location?.trim());

    if (!hasGps && !hasManualLocation) {
      throw new Error(
        'Please detect your location or enter it manually.',
      );
    }

    if (!soilImage) {
      throw new Error(
        'Please upload a soil image.',
      );
    }

    if (!wetBehavior) {
      throw new Error(
        'Please describe how the soil behaves when wet.',
      );
    }

    const formData = new FormData();

    if (hasGps) {
      formData.append(
        'latitude',
        String(latitude),
      );

      formData.append(
        'longitude',
        String(longitude),
      );
    }

    if (
      !hasGps &&
      hasManualLocation
    ) {
      formData.append(
        'location',
        location.trim(),
      );
    }

    formData.append(
      'soil_image',
      soilImage,
    );

    formData.append(
      'wet_behavior',
      wetBehavior,
    );

    formData.append(
      'top_n',
      String(topN),
    );

    const response =
      await axios.post(
        API_URL,
        formData,
        {
          headers: {
            Accept: 'application/json',
          },
        },
      );

    if (
      response.data?.success !== true
    ) {
      throw new Error(
        'Recommendation service returned an unsuccessful response.',
      );
    }

    return response.data;

  } catch (error) {

    const detail =
      error.response?.data?.detail;

    if (Array.isArray(detail)) {
      const message =
        detail
          .map((item) => item.msg)
          .filter(Boolean)
          .join(', ');

      throw new Error(
        message ||
        'The recommendation request was invalid.',
      );
    }

    throw new Error(
      detail ||
      error.message ||
      'Unable to reach the recommendation API.',
    );
  }
}
