import numpy as np
import matplotlib.pyplot as plt
import time as time
import math as m
import os
import csv
import statistics as st

A = np.array([[4, -2, -4],[-2, 10, 5], [-4, 5, 6]])
C = np.array([[6], [-9], [-7]])

def Cholesky(A):

    l = len(A)
    c = len(A)
    L = np.zeros((l,c), dtype=float)

    for i in range(l):
        for k in range(i+1):

            somme = sum(L[i][j]*L[k][j] for j in range(k))
            
            if i==k:
                L[i][k] = m.sqrt(A[k][k]-somme)
            else:
                L[i][k] = ((A[i][k] - somme)/ L[k][k])
    
    Lcopy = np.copy(L)
    Lt = np.transpose(Lcopy)
    # print(L, "\n")
    # print(Lt)
    return L, Lt

def Cholesky2(A):

    l = len(A)
    c = len(A)
    L = np.zeros((l,c), dtype=float)

    for k in range(l):
        somme= 0
        for j in range(k):
            somme += (L[k][j])**2
        L[k][k] = m.sqrt((A[k][k] - somme))
        for i in range(k+1, l):
            somme2 = 0
            for j in range(k):
                somme2 += L[k][j]*L[i][j]
            L[i][k] = (A[i][k] - somme2)/(L[k][k])
    
    Lcopy = np.copy(L)
    Lt = np.transpose(Lcopy)

    return L, Lt

def ResolutionColesky(L, B, Lt):

    l = len(L)
    c = len(L[0])
    yinv = [0]*l    
    Linv = np.fliplr(np.flipud(L))
    Binv = np.flipud(B)
    # print(Linv)
    # print(Binv)
    x = [0]*l

    yinv[l-1] = Binv[l-1][0] / Linv[l-1][l-1]
    for i in range(l-2, -1, -1):
        yinv[i] = Binv[i][0]
        for j in range(i+1, l):
            yinv[i] = yinv[i] - Linv[i][j]*yinv[j]
        yinv[i] = (yinv[i]/Linv[i][i])
    y = np.flipud(yinv)

    x[l-1] = y[l-1] / Lt[l-1][l-1]
    for i in range(l-2, -1, -1):
        x[i] = y[i]
        for j in range(i+1, l):
            x[i] = x[i] - Lt[i][j]*x[j]
        x[i] = (x[i]/Lt[i][i])
    Result = np.asarray(x, dtype=float).reshape(len(L), 1)
    return Result

def npCholesky(A, B):
    L = np.linalg.cholesky(A)
    Lcopy = np.copy(L)
    Lt = np.transpose(Lcopy)

    l = len(L)
    c = len(L[0])
    yinv = [0]*l    
    Linv = np.fliplr(np.flipud(L))
    Binv = np.flipud(B)
    # print(Linv)
    # print(Binv)
    x = [0]*l

    yinv[l-1] = Binv[l-1][0] / Linv[l-1][l-1]
    for i in range(l-2, -1, -1):
        yinv[i] = Binv[i][0]
        for j in range(i+1, l):
            yinv[i] = yinv[i] - Linv[i][j]*yinv[j]
        yinv[i] = (yinv[i]/Linv[i][i])
    y = np.flipud(yinv)

    x[l-1] = y[l-1] / Lt[l-1][l-1]
    for i in range(l-2, -1, -1):
        x[i] = y[i]
        for j in range(i+1, l):
            x[i] = x[i] - Lt[i][j]*x[j]
        x[i] = (x[i]/Lt[i][i])
    Result = np.asarray(x, dtype=float).reshape(len(L), 1)
    return Result

# a = (Cholesky(A))
# print(ResolutionColesky(a[0], C, a[1]))


def ReductionGauss(Aaug):
    """
    Rend la matrice obtenue après l'application de 
    la métode de Gauss à une matrice augmentée de format (n, n+1)
    """

    l = len(Aaug)
    c = len(Aaug[0])
    # print(l, c)

    for i in range(0, l-1):
        # print(p)
        for k in range(i+1, l):
            g = Aaug[k][i] / Aaug[i][i]
            # print(g)
            for j in range(i, l):
                Aaug[k][j] = Aaug[k][j] - (g * Aaug[i][j])
            Aaug[k][c-1] = Aaug[k][c-1] - (g * Aaug[i][c-1])

    return Aaug


# A = np.array([[2, 5, 6, 7], [4, 11, 9, 12], [-2, -8, 7, 3]])
# B = np.array([[1, 1, 1, 1, 1], [2, 4, -3, 2, 1], [-1, -1, 0, -3, 2], [1, -1, 4, 9, -8]])

# print(ReductionGauss(A))
# print(ReductionGauss(B))


def ResolutionSystTriSup(Taug):
    """
    Résolution du système triangulaire obtenue avec la fonction précédente
    """
    l = len(Taug)
    c = len(Taug[0])
    x = [0]*l
    x[l-1] = Taug[l-1][c-1] / Taug[l-1][c-2]
    
    for i in range(l-2, -1, -1):
        x[i] = Taug[i][c-1]
        for j in range(i+1, l):
            x[i] = x[i] - Taug[i][j]*x[j]
        x[i] = (x[i]/Taug[i][i])
    return x



### QUESTION 3 ###

def Gauss(A, B):
    """
    Application de la méthode de Gauss pour résoudre le système (sans pivot)
    """
    if len(A) == len(B):
        x = len(A)
        C = np.concatenate((A, B), axis=1)
    else:
        print("Les matrices n'ont pas le même nombre de ligne")

    Mtrsup = ReductionGauss(C)
    X = ResolutionSystTriSup(Mtrsup)
    Result = np.asarray(X, dtype=float).reshape(x, 1)
    return Result

def DecompositionLU(A):
    """
    Rend la décomposition LU de la matrice A
    """
    U = A
    l = len(A)
    c = len(A[0])
    # print(l, c)
    L = np.eye(l) 
    # L = np.eye(l, l, dtype=int) 
    for i in range(0, l-1):
        # print(p)
        for k in range(i+1, l):
            g = U[k][i] / U[i][i]
            L[k][i] = g
            # print(g)
            for j in range(i, l):
                U[k][j] = U[k][j] - (g * U[i][j])
    
    return L, U

### QUESTION 2 ###

def ResolutionLU(L,B,U):
    """
    Résolution du sytème par la méthode LU
    """
    
    l = len(L)
    c = len(L[0])
    yinv = [0]*l    
    Linv = np.fliplr(np.flipud(L))
    Binv = np.flipud(B)
    # print(Linv)
    # print(Binv)
    x = [0]*l

    yinv[l-1] = Binv[l-1][0] / Linv[l-1][l-1]
    for i in range(l-2, -1, -1):
        yinv[i] = Binv[i][0]
        for j in range(i+1, l):
            yinv[i] = yinv[i] - Linv[i][j]*yinv[j]
        yinv[i] = (yinv[i]/Linv[i][i])
    y = np.flipud(yinv)

    x[l-1] = y[l-1] / U[l-1][l-1]
    for i in range(l-2, -1, -1):
        x[i] = y[i]
        for j in range(i+1, l):
            x[i] = x[i] - U[i][j]*x[j]
        x[i] = (x[i]/U[i][i])
    Result = np.asarray(x, dtype=float).reshape(len(L), 1)
    return Result

def LU(A, B):
    a = DecompositionLU(A)
    b = ResolutionLU(a[1], B, a[0])
    return b

def matrice_alea():
    """
    Donne une liste contenant des listes de matrices aléatoires pour effectuer les tests
    """
    list_cholesky = list()
    list_Gauss = list()
    list_LU = list()
    list_test_LU = list()
    list_numpy_cholesky = list()
    list_numpy = list()
    list_cond = list()
    for i in range(100, 600, 50):
        A = np.random.rand(i,i)
        Acopy = np.copy(A)
        At = np.transpose(Acopy)
        matrice = np.matmul(At,A)
        B = np.copy(matrice)
        C = np.copy(matrice)
        D = np.copy(matrice)
        E = np.copy(matrice)
        F = np.copy(matrice)
        a = np.linalg.cond(matrice)
        list_cond.append(a)
        list_cholesky.append(matrice)
        list_Gauss.append(B)
        list_numpy_cholesky.append(C)
        list_LU.append(D)
        list_test_LU.append(F)
        list_numpy.append(D)
    print("\nConditionnement moyen : ", st.mean(list_cond),"\n")
    return list_cholesky, list_Gauss, list_numpy_cholesky, list_numpy, list_LU, list_test_LU



def vecteur_alea():
    """
    Donne une liste contenant les matrices colonnes aléatoires pour effectuer les tests
    """
    list_vecteur = list()
    for i in range(100, 600, 50):
        B = np.random.rand(i,1)
        list_vecteur.append(B)
    return list_vecteur

