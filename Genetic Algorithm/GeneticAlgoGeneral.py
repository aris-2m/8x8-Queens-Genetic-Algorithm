import sympy
import math
import numpy
import math
from copy import copy


fonction=lambda x: numpy.sin((x*numpy.pi)/(255))


def liste_en_decimale(X):
    binaire=0
    taille=len(X)-1

    for i in X:
        binaire+=(2**taille)*i
        taille=taille-1
    return binaire


def randpop(pop_size=20,plage_valeurs=1):
    population=[]
    for i in range(pop_size):
        nouveau=numpy.random.choice(plage_valeurs+1,8)
        population.append(nouveau)
    return population


def select(pop_size=10,n_parents=2):
    liste_selection=[]
    for i in range(pop_size):
        sous_liste=[]
        for i in range(n_parents):
            a=numpy.random.randint(pop_size)
            sous_liste.append(a)
        liste_selection.append(sous_liste)

    return liste_selection


def elements_croisement_mi(n_parents):

    #n_parents doit etre une liste de parents
    #Pour chaque parent on va créer deux sous chaines moitié moitié et insérer toutes ces moitiés dans une liste unique

    liste=[]
    for parent_i in n_parents:
        longueur=len(parent_i)

        mi=math.floor(longueur/2)

        parent_i1=parent_i[0:mi]
        liste.append(parent_i1)
        parent_i2=parent_i[mi:longueur]
        liste.append(parent_i2)

    #Combinaison

    return liste


X1=[1,1,1,1,1,1,1,1]
X2=[1,0,0,0,1,0,0,0]

l1=elements_croisement_mi([X1,X2])
l2=[]
from itertools import combinations
for i,j in combinations(l1,2):
    l2.append(i+j)



def crossover(parent_a, parent_b):
    fils=[];
    for i in range(len(parent_a)):
        choix=numpy.random.randint(2) #Choisir entre deux nombres 0 et 1

        if (choix==0):
            fils.append(parent_a[i])
        if (choix==1):
            fils.append(parent_b[i])
    return fils



def offspring(list_sel_, population1):
    pop_fille=[]
    for i in list_sel_:
        fille=crossover(population1[i[0]],population1[i[1]])
        pop_fille.append(fille)
    return pop_fille


def mutation(individu):

    individu2=copy(individu)
    #Associer une probabilité à chaque case
    #Une probabilité est un nombre entre 0 et 1 donc 1/p avec p non nul
    #Si p <= seuil on mute la case sinon on le garde

    seuil=1/8
    p=100

    for i in range(0,len(individu2)):
        test="Refaire"
        while(test=="Refaire"):
            deno=numpy.random.randint(p)
            if (deno!=0):
                prob=1/deno
                test="Arret"
                if prob<seuil:
                    individu2[i]=numpy.abs(1-individu2[i])

    return individu2



def mutation_quelconque(individu):

    individu2=copy(individu)
    #Associer une probabilité à chaque case
    #Une probabilité est un nombre entre 0 et 1 donc 1/p avec p non nul
    #Si p <= seuil on mute la case sinon on le garde

    seuil=1/8
    p=100

    for i in range(0,len(individu2)):
        test="Refaire"
        while(test=="Refaire"):
            deno=numpy.random.randint(p)
            if (deno!=0):
                prob=1/deno
                test="Arret"
                if prob<seuil:
                    individu2[i]=numpy.abs(1-individu2[i])

    return individu2



def survival(pop, n_survivors,f=fonction,binaire=True):
    #Conversion de la population en liste:
    pop_list=pop.tolist()

    #Forme decimale des individus
    if(binaire):
        pop_decimale=[]
        for i in pop_list:
            i=liste_en_decimale(i)
            pop_decimale.append(i)
    else:
        pop_decimale=pop_list

    #Calcul de la fitness
    fitness=[]
    for i in pop_decimale:
        fitness.append(f(i))

    #Ordonner les fitness
    fitness1=copy(fitness)
    fitness1.sort(reverse=True) #Procéder comme ça trie par ordre décroissant
    final_survivors_fitness=copy(fitness1[:n_survivors])

    #Ici on veut avoir les indices des éléments triés avec argsort.
    fitness2=copy(fitness)
    sort_croissant_index = numpy.argsort(fitness2).tolist()

    #Mais on a trie par ordre croissant donc on va renverser cette liste pour les avoir en decroissant
    taille = len(sort_croissant_index) - 1
    sort_decroissant_index = []
    while (taille >= 0):
        sort_decroissant_index.append(sort_croissant_index[taille])
        taille = taille - 1

    #Choix des n_survivors premiers avec meilleure fitness ayant leurs indices dans sort_decroissant_index
    premiers_index=copy(sort_decroissant_index[:n_survivors])

    #Les individus ayant premiers_index
    final_survivors=[]
    for i in premiers_index:
        final_survivors.append(pop_list[i])


    return (final_survivors,final_survivors_fitness)



def survival_(pop, n_survivors,f=fonction,binaire=True):
    #Conversion de la population en liste:
    pop_list=pop.tolist()

    #Forme decimale des individus
    if(binaire):
        pop_decimale=[]
        for i in pop_list:
            i=liste_en_decimale(i)
            pop_decimale.append(i)
    else:
        pop_decimale=pop_list

    #Calcul de la fitness
    fitness=[]
    for i in pop_decimale:
        fitness.append(f(i))

    #Ordonner les fitness
    fitness2=copy(fitness)
    fitness2.sort() #Procéder comme ça trie par ordre décroissant
    final2_survivors_fitness=copy(fitness2[:n_survivors])

    #Ici on veut avoir les indices des éléments triés avec argsort.
    fitness2=copy(fitness)
    sort_croissant_index = numpy.argsort(fitness2).tolist()


    #Choix des n_survivors premiers avec meilleure fitness ayant leurs indices dans sort_decroissant_index
    derniers_index=copy(sort_croissant_index[:n_survivors])

    #Les individus ayant premiers_index
    final2_survivors=[]
    for i in derniers_index:
        final2_survivors.append(pop_list[i])


    return (final2_survivors,final2_survivors_fitness)



from scipy.spatial.distance import cdist

def eliminate_duplicates(X):
    D = cdist(X, X)
    D[numpy.triu_indices(len(X))] = numpy.inf
    return numpy.all(D > 1e-32, axis=1)




def main(pop_size=10, nb_parents=2, nb_generations=2, nb_survivants=20, fonction_=fonction, plage_valeurs=1, binaire=True):
    # Creation de la population
    population_=randpop(pop_size,plage_valeurs)

    #Multiplication de la population
    population_generee=copy(population_)
    for i in range(nb_generations):
        #Selection
        list_select=select(pop_size,nb_parents)
        #Croisements
        population_fille_=offspring(list_select, population_generee)
        array_pop_fille_=numpy.array(population_fille_)
        new_population_=numpy.vstack((population_generee, array_pop_fille_))
        #Mutations
        if(binaire):
            population_muted_=mutation(new_population_)
        else:
            population_muted_=new_population_
        #Mise à jour variables
        population_generee=population_muted_
        pop_size=len(population_generee)

    #Les plus forts

    les_survivants=survival(population_generee,nb_survivants,fonction_,binaire)

    les_survivants_=les_survivants[0]
    les_survivants_fitn=les_survivants[1]

    les_survivants_=numpy.array(les_survivants_)
    les_survivants_fitn=numpy.array(les_survivants_fitn)

    pop_sansduplicata_maskbool_=eliminate_duplicates(les_survivants_)

    pop_sansduplicata_=les_survivants_[pop_sansduplicata_maskbool_]

    fitness_pop_sansduplicata_=les_survivants_fitn[pop_sansduplicata_maskbool_]

    #Les plus faibles

    les_survivants2=survival_(population_generee,nb_survivants,fonction_,binaire)

    les_survivants_2=les_survivants2[0]
    les_survivants_fitn2=les_survivants2[1]

    les_survivants_2=numpy.array(les_survivants_2)
    les_survivants_fitn2=numpy.array(les_survivants_fitn2)

    pop_sansduplicata_maskbool_2=eliminate_duplicates(les_survivants_2)

    pop_sansduplicata_2=les_survivants_2[pop_sansduplicata_maskbool_2]

    fitness_pop_sansduplicata_2=les_survivants_fitn2[pop_sansduplicata_maskbool_2]

    return (pop_sansduplicata_,fitness_pop_sansduplicata_,pop_sansduplicata_2,fitness_pop_sansduplicata_2,population_generee)
