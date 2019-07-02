# algo pour traiter les non conformités

- virer les sommets confondus pour ne garder qu'un seul id de sommet (peut-être n'est-ce pas nécessaire)
- enregistrer la description des faces par leurs arêtes
- enregistrer la description des cellules par leurs faces
- repérer toutes les paires de piliers qui présentent des non-conformité
- pour chaque pilier présentant des non conformités
    * réordonner les sommets sur les piliers (du bas vers le haut par ex)
    * décrire les anciennes aretes subdivisées (cf. API)
- pour chaque paire de pilier
    * trianguler
    * ajouter les points d'intersection interne comme nouveaux noeuds/vertices
    * creer les nouvelles faces issues de la triangulation
    * decrire les anciennes faces (subdivisées ou non) en utilisant les nouvelles faces créées
- faire une passe finale pour 
    * remplacer dans toutes les faces les arêtes subdivisées
    * éliminer les arêtes inutiles et renuméroter (id) les arêtes
    * remplacer les arêtes dans la description des faces par leur id
    * remplacer dans toutes les cellules les faces substituées par d'autre(s)
    * éliminer les arêtes inutiles et renuméroter (id) les faces
    * remplacer les faces dans la description des cellules par leur nouvelle id
    

# génération de maillages Petrel de test

On assemble sur une grille des cubes obtenus par translation/homotéthie d'un 
cube unité. Chaque cube est décrit par les coordonnées de ses 8 sommets, sous
la forme d'un numpy array de shape (8, 3). Les opérations de translation sont ainsi
immédiates puisqu'il suffit d'ajouter un vecteur au tableau de noeuds.
La *grille* est décrite par un tableau de taille cubes=(ncx, ncy, ncz, 8, 3) avec
ncx, ncy, ncz respectivement le nombre de cellules suivant Ox, Oy, Oz.

On a directement le tableau zcoord qui correspond à cubes[...,2].
On peut verifier que les piliers sont bien rectilignes puis obtenir leurs mins
max à coup numpy.argmin / numpy.argmax.


