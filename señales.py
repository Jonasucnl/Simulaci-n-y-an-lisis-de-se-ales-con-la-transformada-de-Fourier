import numpy as np
import matplotlib.pyplot as plt

# Parámetros generales
N = 1000
t = np.linspace(-5, 5, N)
dt = t[1] - t[0]
freqs = np.fft.fftfreq(N, dt)

# ==========================================
# 1. VERIFICACIÓN DE LA PROPIEDAD DE LINEALIDAD
# ==========================================
# Si y(t) = a*x1(t) + b*x2(t), entonces Y(f) = a*X1(f) + b*X2(f)
x1 = np.where((t >= -1) & (t <= 1), 1.0, 0.0)  # Pulso rectangular
x2 = np.sin(2 * np.pi * 1.0 * t)                # Función senoidal

a, b = 2.5, 1.5
y_tiempo = a * x1 + b * x2

# Transformada de la combinación lineal en el tiempo
Y_tiempo_fft = np.fft.fft(y_tiempo)

# Combinación lineal de las transformadas individuales
X1_fft = np.fft.fft(x1)
X2_fft = np.fft.fft(x2)
Y_esperada_fft = a * X1_fft + b * X2_fft

# Comprobación numérica exacta
es_lineal = np.allclose(Y_tiempo_fft, Y_esperada_fft, atol=1e-10)
print(f"-> Propiedad de Linealidad verificada: {es_lineal}")


# ==========================================
# 2. VERIFICACIÓN DE DESPLAZAMIENTO EN EL TIEMPO
# ==========================================
# Si x(t) -> X(f), entonces x(t - t0) -> X(f) * e^(-j * 2*pi*f*t0)
# Esto implica que la MAGNITUD del espectro no cambia, solo la fase.
t0 = 1.5  # Retardo en el tiempo
x_original = np.where((t >= -1) & (t <= 1), 1.0, 0.0)
# Señal desplazada (asegurando mantener el mismo vector de tiempo)
x_desplazada = np.where(((t - t0) >= -1) & ((t - t0) <= 1), 1.0, 0.0)

X_orig = np.fft.fft(x_original)
X_desp = np.fft.fft(x_desplazada)

# La magnitud debe ser idéntica
magnitud_igual = np.allclose(np.abs(X_orig), np.abs(X_desp), atol=1e-3)
print(f"-> Propiedad de Desplazamiento (conservación de magnitud) verificada: {magnitud_igual}")


# ==========================================
# 3. VERIFICACIÓN DE ESCALAMIENTO TEMPORAL
# ==========================================
# Si x(t) -> X(f), entonces x(a*t) -> (1 / |a|) * X(f / a)
# Un factor 'a > 1' comprime la señal en el tiempo y expande el espectro en frecuencia.
a_factor = 2.0
x_escalada_t = np.where(((t * a_factor) >= -1) & ((t * a_factor) <= 1), 1.0, 0.0)

X_orig_ft = np.fft.fft(x_original)
X_esc_ft = np.fft.fft(x_escalada_t)


# ==========================================
# GRÁFICAS DE COMPROBACIÓN
# ==========================================
plt.figure(figsize=(14, 10))

# Gráfica Linealidad
plt.subplot(3, 1, 1)
plt.plot(freqs, np.abs(Y_tiempo_fft)/N, label='FFT de a·x1 + b·x2', color='purple', linewidth=2)
plt.plot(freqs, np.abs(Y_esperada_fft)/N, label='a·FFT(x1) + b·FFT(x2)', color='orange', linestyle='--', linewidth=1.5)
plt.title('Verificación de Linealidad (Magnitudes idénticas)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('|Y(f)|')
plt.legend()
plt.grid(True)
plt.xlim(-10, 10)

# Gráfica Desplazamiento en el Tiempo (Magnitudes iguales)
plt.subplot(3, 1, 2)
plt.plot(freqs, np.abs(np.fft.fftshift(X_orig))/N, label='Magnitud X(f) Original', color='blue', linewidth=2)
plt.plot(freqs, np.abs(np.fft.fftshift(X_desp))/N, label=f'Magnitud X(f) Desplazada (t0={t0})', color='red', linestyle='--', linewidth=1.5)
plt.title('Verificación de Desplazamiento en el Tiempo (La magnitud no se altera)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('|X(f)|')
plt.legend()
plt.grid(True)
plt.xlim(-10, 10)

# Gráfica Escalamiento Temporal (Ensanchamiento en frecuencia)
plt.subplot(3, 1, 3)
plt.plot(freqs, np.abs(np.fft.fftshift(X_orig_ft))/N, label='Espectro Original X(f)', color='green', linewidth=1.5)
plt.plot(freqs, np.abs(np.fft.fftshift(X_esc_ft))/N, label='Espectro Escalado x(a*t)', color='brown', linewidth=1.5)
plt.title('Verificación de Escalamiento Temporal (Compresión en tiempo = Expansión en frecuencia)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('|X(f)|')
plt.legend()
plt.grid(True)
plt.xlim(-15, 15)

plt.tight_layout()
plt.show()