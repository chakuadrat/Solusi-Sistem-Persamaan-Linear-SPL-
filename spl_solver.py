import numpy as np

def main():
  """Fungsi utama untuk menjalankan program."""
  while True:
    print("\nNisrina Azka Salsabila")
    print("21120122130057")
    print ("Metode Numerik - Kelas C")
    print("Teknik Komputer")
    print("\nPenyelesaian Sistem Persamaan Linear (SPL)")
    print("\nSelamat datang! Silahkan pilih penyelesaian yang anda inginkan pada menu dibawah ini:")
    print("1. Matriks Balikan")
    print("2. Dekomposisi LU Gauss")
    print("3. Dekomposisi Crout")
    print("4. Keluar")

    pilihan = int(input("Masukkan pilihan Anda (1-4): "))

    if pilihan == 1:
      print("\n--- Metode Matriks Balikan ---")
      sistem_matriks_balikan()
    elif pilihan == 2:
      print("\n--- Metode Dekomposisi LU Gauss ---")
      try:
        sistem_dekomposisi_lu_gauss()
      except ValueError as e:
        print(e)
    elif pilihan == 3:
      print("\n--- Metode Dekomposisi Crout ---")
      try:
        sistem_dekomposisi_crout()
      except ValueError as e:
        print(e)
    elif pilihan == 4:
      break
    else:
      print("Pilihan tidak valid. Silakan coba lagi.")

def sistem_matriks_balikan():
  """Fungsi untuk menyelesaikan SPL dengan matriks balikan."""
  n = int(input("Masukkan jumlah variabel (n): "))
  A = np.zeros((n, n))
  b = np.zeros(n)

  # Memasukkan elemen matriks A dan vektor b
  for i in range(n):
    for j in range(n):
      A[i, j] = float(input(f"Masukkan elemen A[{i+1},{j+1}]: "))
    b[i] = float(input(f"Masukkan elemen b[{i+1}]: "))

  # Mencari matriks balikan
  A_inv = np.linalg.inv(A)

  # Mencari solusi
  x = np.dot(A_inv, b)

  # Menampilkan hasil
  print("\nSolusi:")
  for i in range(n):
    print(f"x[{i+1}] = {x[i]}")

def sistem_dekomposisi_lu_gauss():
  """Fungsi untuk menyelesaikan SPL dengan dekomposisi LU Gauss."""
  n = int(input("Masukkan jumlah variabel (n): "))
  A = np.zeros((n, n))
  b = np.zeros(n)

  # Memasukkan elemen matriks A dan vektor b
  for i in range(n):
    for j in range(n):
      A[i, j] = float(input(f"Masukkan elemen A[{i+1},{j+1}]: "))
    b[i] = float(input(f"Masukkan elemen b[{i+1}]: "))

  # Dekomposisi LU
  L, U = gauss_decomposition(A)

  # Solusi dari Ly = b (metode substitusi maju)
  y = forward_substitution(L, b)

  # Solusi dari Ux = y (metode substitusi mundur)
  x = backward_substitution(U, y)

  # Menampilkan hasil
  print("\nSolusi:")
  for i in range(n):
    print(f"x[{i+1}] = {x[i]}")

def sistem_dekomposisi_crout():
  """Fungsi untuk menyelesaikan SPL dengan dekomposisi Crout."""
  n = int(input("Masukkan jumlah variabel (n): "))
  A = np.zeros((n, n))
  b = np.zeros(n)

  # Memasukkan elemen matriks A dan vektor b
  for i in range(n):
    for j in range(n):
      A[i, j] = float(input(f"Masukkan elemen A[{i+1},{j+1}]: "))
    b[i] = float(input(f"Masukkan elemen b[{i+1}]: "))

  # Dekomposisi Crout
  L, U = crout_decomposition(A)

  # Solusi dari Ly = b (metode substitusi maju)
  y = forward_substitution(L, b)

  # Solusi dari Ux = y (metode substitusi mundur)
  x = backward_substitution(U, y)

  # Menampilkan hasil
  print("\nSolusi:")
  for i in range(n):
    print(f"x[{i+1}] = {x[i]}")

def gauss_decomposition(A):
  """Melakukan dekomposisi LU Gauss dari matriks A."""
  n = len(A)
  L = np.eye(n)
  U = A.copy()

  for k in range(n-1):
    if U[k, k] == 0:
      raise ValueError("Matriks tidak dapat didekomposisi dengan metode LU Gauss karena terdapat elemen diagonal utama yang nol.")
    for i in range(k+1, n):
      factor = U[i, k] / U[k, k]
      L[i, k] = factor
      U[i, k:] -= factor * U[k, k:]

  if U[n-1, n-1] == 0:
    raise ValueError("Matriks tidak dapat didekomposisi dengan metode LU Gauss karena terdapat elemen diagonal utama yang nol.")

  return L, U

def crout_decomposition(A):
  """Melakukan dekomposisi Crout dari matriks A."""
  n = len(A)
  L = np.zeros((n, n))
  U = np.zeros((n, n))

  for j in range(n):
    U[j, j] = 1  # Matriks U memiliki diagonal utama yang semuanya 1

    for i in range(j, n):
      sum_l = sum(L[i, k] * U[k, j] for k in range(i))
      L[i, j] = A[i, j] - sum_l

    for i in range(j, n):
      sum_u = sum(L[j, k] * U[k, i] for k in range(j))
      if L[j, j] == 0:
        raise ValueError("Matriks tidak dapat didekomposisi dengan metode Crout karena terdapat elemen diagonal utama pada L yang nol.")
      U[j, i] = (A[j, i] - sum_u) / L[j, j]

  return L, U

def forward_substitution(L, b):
  """Metode substitusi maju untuk mencari solusi Ly = b."""
  n = len(b)
  y = np.zeros(n)

  for i in range(n):
    y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

  return y

def backward_substitution(U, y):
  """Metode substitusi mundur untuk mencari solusi Ux = y."""
  n = len(y)
  x = np.zeros(n)

  for i in range(n - 1, -1, -1):
    x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

  return x

if __name__ == "__main__":
    main(
