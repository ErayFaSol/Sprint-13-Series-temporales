import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

if not os.path.exists('images'):
    os.makedirs('images')

def plot_orders_per_hour(day, data_resampled, filename='orders_per_hour.png'):
    """Gráfica de pedidos de taxis por hora para un día específico."""
    
    specific_day = day
    data_day = data_resampled.loc[specific_day]
    
    plt.figure(figsize=(14, 8))
    plt.plot(data_day.index, data_day['num_orders'], marker='o')
    plt.title(f'Número de Pedidos de Taxis por Hora')
    plt.xlabel('Hora del Día')
    plt.ylabel('Número de Pedidos')
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.gcf().autofmt_xdate()
    plt.savefig('images/' + filename)  # Guardar gráfica como imagen
    plt.close()  # Cerrar la figura para evitar que se mantenga en memoria

def plot_orders_per_day(s_date, e_date, data_resampled, filename='orders_per_day.png'):
    """Gráfica de pedidos de taxis por día durante un mes."""
    
    start_date = s_date
    end_date = e_date
    data_month = data_resampled.loc[start_date:end_date]
    data_daily = data_month.resample('D').sum()
    
    plt.figure(figsize=(14, 8))
    plt.plot(data_daily.index, data_daily['num_orders'], marker='o')
    plt.title(f'Número de Pedidos de Taxis por Día (Marzo 2018)')
    plt.xlabel('Día')
    plt.ylabel('Número de Pedidos')
    plt.grid(True)
    plt.savefig('images/' + filename)  # Guardar gráfica como imagen
    plt.close()  # Cerrar la figura para evitar que se mantenga en memoria

def plot_smoothed_orders(data_week, smoothed_data, window=24, filename='smoothed_orders.png'):
    """Gráfica de media móvil (rolling window) del número de pedidos de taxis por hora."""
    plt.figure(figsize=(14, 10))

    # Serie original
    ax1 = plt.subplot(2, 1, 1)
    plt.plot(data_week.index, data_week['num_orders'], label='Original', color='blue', alpha=0.6)
    plt.title('Número de Pedidos de Taxis por Hora (Primera Semana de Marzo 2018)')
    plt.ylabel('Número de Pedidos')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%H'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Media móvil
    ax2 = plt.subplot(2, 1, 2)
    plt.plot(data_week.index, smoothed_data, label=f'Media Móvil ({window} horas)', color='red', linewidth=2)
    plt.title('Media Móvil del Número de Pedidos de Taxis por Hora')
    plt.xlabel('Fecha y Hora')
    plt.ylabel('Número de Pedidos')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%H'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig('images/' + filename)  # Guardar gráfica como imagen
    plt.close()  # Cerrar la figura para evitar que se mantenga en memoria

def plot_seasonal_decomposition(result, filename='seasonal_decomposition.png'):
    """Descomposición de la serie temporal en tendencia, estacionalidad y residuos."""
    plt.figure(figsize=(14, 12))

    # Componente de tendencia
    plt.subplot(3, 1, 1)
    plt.plot(result.trend, label='Tendencia')
    plt.title('Tendencia de la Serie Temporal')
    plt.ylabel('Número de Pedidos')
    plt.legend()
    plt.grid(True)

    # Componente estacional
    plt.subplot(3, 1, 2)
    plt.plot(result.seasonal, label='Estacionalidad', color='orange')
    plt.title('Estacionalidad de la Serie Temporal')
    plt.ylabel('Número de Pedidos')
    plt.legend()
    plt.grid(True)

    # Componente residual
    plt.subplot(3, 1, 3)
    plt.plot(result.resid, label='Residuos', color='green')
    plt.title('Residuos de la Serie Temporal')
    plt.ylabel('Número de Pedidos')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('images/' + filename)  # Guardar gráfica como imagen
    plt.close()  # Cerrar la figura para evitar que se mantenga en memoria

def plot_autocorrelation(data_resampled, filename='autocorrelation.png'):
    """Gráfica de autocorrelación de la serie temporal."""
    from pandas.plotting import autocorrelation_plot
    plt.figure(figsize=(14, 8))
    autocorrelation_plot(data_resampled['num_orders'])
    plt.title('Autocorrelación del Número de Pedidos')
    plt.xlabel('Retraso (Lag)')
    plt.ylabel('Autocorrelación')
    plt.grid(True)
    plt.savefig('images/' + filename)  # Guardar gráfica como imagen
    plt.close()  # Cerrar la figura para evitar que se mantenga en memoria
