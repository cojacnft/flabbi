import { configureStore } from '@reduxjs/toolkit';
import metricsReducer from './slices/metricsSlice';
import opportunitiesReducer from './slices/opportunitiesSlice';
import alertsReducer from './slices/alertsSlice';
import systemReducer from './slices/systemSlice';

export const store = configureStore({
  reducer: {
    metrics: metricsReducer,
    opportunities: opportunitiesReducer,
    alerts: alertsReducer,
    system: systemReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: false,
    }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;