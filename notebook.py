import marimo

__generated_with = "0.9.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md("""#3D Geometry File Formats""")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## About STL

        STL is a simple file format which describes 3D objects as a collection of triangles.
        The acronym STL stands for "Simple Triangle Language", "Standard Tesselation Language" or "STereoLitography"[^1].

        [^1]: STL was invented for – and is still widely used – for 3D printing.
        """
    )
    return


@app.cell
def __(mo, show):
    mo.show_code(show("data/teapot.stl", theta=45.0, phi=30.0, scale=2))
    return


@app.cell
def __(mo, show):
    with open("data/teapot.stl", mode="rt", encoding="utf-8") as _file:
        teapot_stl = _file.read()

    teapot_stl_excerpt = teapot_stl[:723] + "..." + teapot_stl[-366:]

    show("data/teapot.stl", theta=45.0, phi=30.0, scale=2)

    mo.md(
        f"""
    ## STL ASCII Format

    The `data/teapot.stl` file provides an example of the STL ASCII format. It is quite large (more than 60000 lines) and looks like that:
    """
    +
    f"""```
    {teapot_stl_excerpt}
    ```
    """
    +

    """
    """
    )
    return teapot_stl, teapot_stl_excerpt


@app.cell
def __(mo, show):
    mo.md(f"""

      - Study the [{mo.icon("mdi:wikipedia")} STL (file format)](https://en.wikipedia.org/wiki/STL_(file_format)) page (or other online references) to become familiar the format.

      - Create a STL ASCII file `"data/cube.stl"` that represents a cube of unit length  
        (💡 in the simplest version, you will need 12 different facets).

      - Display the result with the function `show` (make sure to check different angles).
    """)

    sommets= {
        'A': (0, 0, 0), 'B': (1, 0, 0), 'C': (0, 1, 0), 'D': (1, 1, 0),  # Sommets de la face inférieure
        'E': (0, 0, 1), 'F': (1, 0, 1), 'G': (0, 1, 1), 'H': (1, 1, 1)   # Sommets de la face supérieure
    } #coordonnés des sommets du cube
    # J'ai aussi donné des noms aux sommets et défini à l'avance les décomPlacements des faces en triangles 

    normales= [(0,0,1),(0,0,-1),(0,1,0),(0,-1,0),(1,0,0),(-1,0,0)] # Pour ce cube on connait les normales aux faces, pas besoin de les calculer 

    triangles_normales= [
        # Face inférieure de normale (0, 0, -1)
        (('A', 'B', 'C'), (0, 0, -1)), (('B', 'D', 'C'), (0, 0, -1)),
        # Face supérieure de normale (0, 0, 1)
        (('E', 'F', 'G'), (0, 0, 1)), (('F', 'H', 'G'), (0, 0, 1)),
        # Face avant de normale (0, 1, 0)
        (('A', 'C', 'E'), (0, 1, 0)), (('C', 'G', 'E'), (0, 1, 0)),
        # Face arrière de normale (0, -1, 0)
        (('B', 'F', 'D'), (0, -1, 0)), (('D', 'F', 'H'), (0, -1, 0)),
        # Face gauche de normale (-1, 0, 0)
        (('A', 'E', 'B'), (-1, 0, 0)), (('B', 'E', 'F'), (-1, 0, 0)),
        # Face droite de normale (1, 0, 0)
        (('C', 'D', 'G'), (1, 0, 0)), (('D', 'H', 'G'), (1, 0, 0))
                   ]
    #J'ai défini tous les points, toutes les faces et les aient associées à leurs normales 
    with open("data/cube.stl", mode="w", encoding="utf-8") as cube_stl: #on passe en mode w pour écrire dans le fichier 
        cube_stl.write("solid\n")
        for (triangle,normale) in triangles_normales:
            v1, v2, v3 = sommets[triangle[0]], sommets[triangle[1]], sommets[triangle[2]] #on récupère les sommets du triangle 
            cube_stl.write(f"  facet normal {normale[0]} {normale[1]} {normale[2]}\n")
            cube_stl.write("    outer loop\n")
            cube_stl.write(f"      vertex {v1[0]} {v1[1]} {v1[2]}\n") #Coordonnées des sommets 
            cube_stl.write(f"      vertex {v2[0]} {v2[1]} {v2[2]}\n")
            cube_stl.write(f"      vertex {v3[0]} {v3[1]} {v3[2]}\n")
            cube_stl.write("    endloop\n")
            cube_stl.write("  endfacet\n")
        cube_stl.write("endsolid cube\n")


    show("data/cube.stl", theta=30, phi=50, scale=1)
    return (
        cube_stl,
        normale,
        normales,
        sommets,
        triangle,
        triangles_normales,
        v1,
        v2,
        v3,
    )


@app.cell
def __(mo):
    mo.md(r"""## STL & NumPy""")
    return


@app.cell
def __(mo, np):
    mo.md(rf"""

    ### NumPy to STL

    Implement the following function:

    ```python
    def make_STL(triangles, normals=None, name=""):
        pass # 🚧 TODO!
    ```

    #### Parameters

      - `triangles` is a NumPy array of shape `(n, 3, 3)` and data type `np.float32`,
         which represents a sequence of `n` triangles (`triangles[i, j, k]` represents 
         is the `k`th coordinate of the `j`th point of the `i`th triangle)

      - `normals` is a NumPy array of shape `(n, 3)` and data type `np.float32`;
         `normals[i]` represents the outer unit normal to the `i`th facet.
         If `normals` is not specified, it should be computed from `triangles` using the 
         [{mo.icon("mdi:wikipedia")} right-hand rule](https://en.wikipedia.org/wiki/Right-hand_rule).

      - `name` is the (optional) solid name embedded in the STL ASCII file.

    #### Returns

      - The STL ASCII description of the solid as a string.

    #### Example

    Given the two triangles that make up a flat square:

    ```python

    square_triangles = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    ```

    then printing `make_STL(square_triangles, name="square")` yields
    ```
    solid square
      facet normal 0.0 0.0 1.0
        outer loop
          vertex 0.0 0.0 0.0
          vertex 1.0 0.0 0.0
          vertex 0.0 1.0 0.0
        endloop
      endfacet
      facet normal 0.0 0.0 1.0
        outer loop
          vertex 1.0 1.0 0.0
          vertex 0.0 1.0 0.0
          vertex 1.0 0.0 0.0
        endloop
      endfacet
    endsolid square
    ```

    """)



    def make_STL(triangles, normals=None):
        print(triangles.shape)

        n= triangles.shape[0]

        if normals is None: #On calcule les normes si elles ne sont pas données 
            normals = np.zeros((n, 3))  #initialisation
            for i in range(n):# triangles[i,j,k] avec 0<=i<=n-1 / 0<=j<=2 (3 sommets) / 0<=k<=2 (3coordonnées)
                v1 = triangles[i, 1, :] - triangles[i, 0, :] 
                v2 = triangles[i, 2, :] - triangles[i, 0, :] #v1,v2 sont 2 arrêtes du triangle. On a besoin que de deux arrêtes pour trouver le vecteur normal
                normal = np.cross(v1, v2) #produit vectoriel pour avoir un vecteur orthogonal
                normals[i]=normal  
        stl_str = f"solid\n"

        #print(v1,v2,normal)

        for i in range(n):
            stl_str += f"  facet normal {normals[i, 0]} {normals[i, 1]} {normals[i, 2]}\n" #Pour chaque triangle on a une normale 
            stl_str += "    outer loop\n"
            for j in range(3):
                stl_str += f"      vertex {triangles[i, j, 0]} {triangles[i, j, 1]} {triangles[i, j, 2]}\n" #pour chaque triangle on a 3 sommets
            stl_str += "    endloop\n"
            stl_str += "  endfacet\n"

        stl_str += f"endsolid\n"

        return stl_str

    #Vérification avec l'exemple donné 
    make_STL(np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ],),normals=None)
    return (make_STL,)


@app.cell
def __(mo):
    mo.md(
        """
        ### STL to NumPy

        Implement a `tokenize` function


        ```python
        def tokenize(stl):
            pass # 🚧 TODO!
        ```


        that is consistent with the following documentation:


        #### Parameters

          - `stl`: a Python string that represents a STL ASCII model.

        #### Returns

          - `tokens`: a list of STL keywords (`solid`, `facet`, etc.) and `np.float32` numbers.

        #### Example

        For the ASCII representation the square `data/square.stl`, printing the tokens with

        ```python
        with open("data/square.stl", mode="rt", encoding="us-ascii") as square_file:
            square_stl = square_file.read()
        tokens = tokenize(square_stl)
        print(tokens)
        ```

        yields

        ```python
        ['solid', 'square', 'facet', 'normal', np.float32(0.0), np.float32(0.0), np.float32(1.0), 'outer', 'loop', 'vertex', np.float32(0.0), np.float32(0.0), np.float32(0.0), 'vertex', np.float32(1.0), np.float32(0.0), np.float32(0.0), 'vertex', np.float32(0.0), np.float32(1.0), np.float32(0.0), 'endloop', 'endfacet', 'facet', 'normal', np.float32(0.0), np.float32(0.0), np.float32(1.0), 'outer', 'loop', 'vertex', np.float32(1.0), np.float32(1.0), np.float32(0.0), 'vertex', np.float32(0.0), np.float32(1.0), np.float32(0.0), 'vertex', np.float32(1.0), np.float32(0.0), np.float32(0.0), 'endloop', 'endfacet', 'endsolid', 'square']
        ```
        """
    )
    return


@app.cell
def __(np):
    #Je fais une fonction qui identifie le type (str ou float) des termes 
    def est_float(str):
        try :
            float(str)
            return True
        except ValueError:
            return False

    def tokenize(stl):
        tokens = []
        for line in stl.splitlines():  #line prend comme valeurs la chaine de caractère de tous les mots sur une ligne de stl
            for word in line.split(): # word prend comme valeur successivement les mots sur line 
                if est_float(word):
                    tokens.append(np.float32(word))
                else:
                    tokens.append(word)
        return tokens
    #Si float(word) est un flottant il est ajouté en tant que tel sinon il est ajouté comme word


    with open("data/square.stl", mode="rt", encoding="us-ascii") as square_file2:
        square_stl2 = square_file2.read()
        tokens90 = tokenize(square_stl2)
        print(tokens90)
    return est_float, square_file2, square_stl2, tokenize, tokens90


@app.cell
def __(mo):
    mo.md(
        """
        Implement a `parse` function


        ```python
        def parse(tokens):
            pass # 🚧 TODO!
        ```

        that is consistent with the following documentation:


        #### Parameters

          - `tokens`: a list of tokens

        #### Returns

        A `triangles, normals, name` triple where

          - `triangles`: a `(n, 3, 3)` NumPy array with data type `np.float32`,

          - `normals`: a `(n, 3)` NumPy array with data type `np.float32`,

          - `name`: a Python string.

        #### Example

        For the ASCII representation `square_stl` of the square,
        tokenizing then parsing

        ```python
        with open("data/square.stl", mode="rt", encoding="us-ascii") as square_file:
            square_stl = square_file.read()
        tokens = tokenize(square_stl)
        triangles, normals, name = parse(tokens)
        print(repr(triangles))
        print(repr(normals))
        print(repr(name))
        ```

        yields

        ```python
        array([[[0., 0., 0.],
                [1., 0., 0.],
                [0., 1., 0.]],

               [[1., 1., 0.],
                [0., 1., 0.],
                [1., 0., 0.]]], dtype=float32)
        array([[0., 0., 1.],
               [0., 0., 1.]], dtype=float32)
        'square'
        ```
        """
    )
    return


@app.cell
def __(np, tokenize):
    def parse(tokens):
        
        triangles = []
        normals = []
        name = None
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            if token == "solid":
                name = tokens[i + 1]  #Si le token est "solid" cela signifie que le token suivant contient le nom (document STL)
                i += 2 #parce que les 2 prochains auraient été les tokens "solid" et "name"
            elif token == "facet":
                normal = [float(tokens[i + 2]), float(tokens[i + 3]), float(tokens[i + 4])] # le ieme aurait été le token "facet" et le i+1 aurait été "normal"
                normals.append(normal)
                i += 5 #On passe les tokens que l'on vient de traiter
            elif token == "outer":
                i += 2  #On passe les 2 tokens "outer" et "loop"

                vertex1 = [float(tokens[i + 1]), float(tokens[i + 2]), float(tokens[i + 3])] #Le ieme aurait été vertex
                vertex2 = [float(tokens[i + 5]), float(tokens[i + 6]), float(tokens[i + 7])] #le i+4 ieme aurait été vertex
                vertex3 = [float(tokens[i + 9]), float(tokens[i + 10]), float(tokens[i + 11])] #le i+8ieme aurait été vertex
                triangles.append([vertex1, vertex2, vertex3])
                i += 12 #on passe toute la section que l'on vient de traiter (4 tokens x 3 = 12 )
            elif token == "endfacet":
                i += 1  
            elif token == "endsolid":
                i += 1  
            else:
                i += 1 #On passe au token suivant 


        triangles = np.array(triangles)
        normals = np.array(normals)
        return triangles, normals, name

    with open("data/square.stl", mode="rt", encoding="us-ascii") as square_file:
        square_stl = square_file.read()
    tokens = tokenize(square_stl)
    triangles, normals, name = parse(tokens)
    print(repr(triangles))
    print(repr(normals))
    print(repr(name))
    return name, normals, parse, square_file, square_stl, tokens, triangles


@app.cell
def __(mo, np, parse, tokenize):
    mo.md(
        rf"""
    ## Rules & Diagnostics



        Make diagnostic functions that check whether a STL model satisfies the following rules

          - **Positive octant rule.** All vertex coordinates are non-negative.

          - **Orientation rule.** All normals are (approximately) unit vectors and follow the [{mo.icon("mdi:wikipedia")} right-hand rule](https://en.wikipedia.org/wiki/Right-hand_rule).

          - **Shared edge rule.** Each triangle edge appears exactly twice.

          - **Ascending rule.** the z-coordinates of (the barycenter of) each triangle are a non-decreasing sequence.

    When the rule is broken, make sure to display some sensible quantitative measure of the violation (in %).

    For the record, the `data/teapot.STL` file:

      - 🔴 does not obey the positive octant rule,
      - 🟠 almost obeys the orientation rule, 
      - 🟢 obeys the shared edge rule,
      - 🔴 does not obey the ascending rule.

    Check that your `data/cube.stl` file does follow all these rules, or modify it accordingly!

    """
    )

    def positive_octant_rule(triangles): #triangles est un tableau numpy de taille (n,3,3) avec n le nombre de triangles, de 3 sommets, avec 3 coordonnées (x,y,z)
        coords_tot=triangles.size # Le nombre de coordonnées est le nombre d'éléments dans le tableau
        coords_neg= np.sum(triangles<0)
        erreur= coords_neg/coords_tot*100
        return erreur 

    with open("data/cube.stl", mode="rt", encoding="us-ascii") as cube2_file:
        cube2_stl = cube2_file.read()
    tokens2 = tokenize(cube2_stl)
    triangles2, normales2, name2 = parse(tokens2)
    print (positive_octant_rule(triangles2))
    #Mon cube vérifie la règle (si on se réfère au code de construction du cube, j'ai donné que des coordonnées positives pour les sommets du cube)

    with open("data/teapot.stl", mode="rt", encoding="us-ascii") as tea_file:
        tea_stl = tea_file.read()
    tokens3 = tokenize(tea_stl)
    triangles3, normales3, name3 = parse(tokens3)
    print (positive_octant_rule(triangles3))
    # Erreure d'environ 32% pour la teapot
    return (
        cube2_file,
        cube2_stl,
        name2,
        name3,
        normales2,
        normales3,
        positive_octant_rule,
        tea_file,
        tea_stl,
        tokens2,
        tokens3,
        triangles2,
        triangles3,
    )


@app.cell
def __(np, parse, tokenize):
    def ascending_rule(triangles):
        barycentres = np.mean(triangles, axis=1) #Tableau où il n'y a que les barycentres de chaque triangle (barycentre= moyenne arithmétique des coord x,y,z)
        z_coords = barycentres[:, 2]  # dans le tableau barycentres on récupère la liste des coordonnées z 
        #print(z_coords)
        compteur = 0
        for i in range(1, len(z_coords)): 
            if z_coords[i-1] > z_coords[i]:
                compteur += 1
        erreur= compteur/len(z_coords)*100
        return erreur

    with open("data/cube.stl", mode="rt", encoding="us-ascii") as cube4_file:
        cube4_stl = cube4_file.read()
    tokens4 = tokenize(cube4_stl)
    triangles4, normals4, name4 = parse(tokens4)
    print (ascending_rule(triangles4))
    #erreure d'environ 33% pour mon cube 

    with open("data/teapot.stl", mode="rt", encoding="us-ascii") as tea2_file:
        tea2_stl = tea2_file.read()
    tokens5 = tokenize(tea2_stl)
    triangles5, normales5, name5 = parse(tokens5)
    print (ascending_rule(triangles5))

    #Erreure d'environ 35% pour la teapot
    return (
        ascending_rule,
        cube4_file,
        cube4_stl,
        name4,
        name5,
        normales5,
        normals4,
        tea2_file,
        tea2_stl,
        tokens4,
        tokens5,
        triangles4,
        triangles5,
    )


@app.cell
def __(parse, tokenize):
    def edge_rule(triangles):
        edges = []
        for triangle in triangles:
            edges.append(tuple(sorted((tuple(triangle[0]), tuple(triangle[1]))))) #sorted permet de ne pas avoir une orientation pour les arrêtes, l'arrête AB est la même que BA
    #On passe tout en tuple pour avoir l'immuabilité 
            edges.append(tuple(sorted((tuple(triangle[1]), tuple(triangle[2])))))
            edges.append(tuple(sorted((tuple(triangle[2]), tuple(triangle[0])))))

        from collections import Counter
        edge_counts = Counter(edges) #Counter va compter toutes les occurences des éléments de edges, edge_counts est un dictionnaire où les clés sont les éléments (donc les arêtes = paires de points triées) et les valeurs le nombre d'occurence
        print(edge_counts)
        nb_erreur = sum(occurence != 2 for occurence in edge_counts.values())

        # Calcul de l'erreur commise 
        total_edges = len(edges)
        erreur = (nb_erreur / total_edges) * 100
        return erreur

    with open("data/cube.stl", mode="rt", encoding="us-ascii") as cube11_file:
        cube11_stl = cube11_file.read()
    tokens11 = tokenize(cube11_stl)
    triangles11, normals11, name11 = parse(tokens11)
    print (edge_rule(triangles11))
    #Mon cube respecte la règle

    with open("data/teapot.stl", mode="rt", encoding="us-ascii") as tea12_file:
        tea12_stl = tea12_file.read()
    tokens12 = tokenize(tea12_stl)
    triangles12, normales12, name12 = parse(tokens12)
    print (edge_rule(triangles12))
    # La teapot vérifie bien la règle
    return (
        cube11_file,
        cube11_stl,
        edge_rule,
        name11,
        name12,
        normales12,
        normals11,
        tea12_file,
        tea12_stl,
        tokens11,
        tokens12,
        triangles11,
        triangles12,
    )


@app.cell
def __(cube2_stl, np, parse, tokenize):
    def orientation_rule(triangles, normales, tolerance=1e-6):
        
        #Vérification des normales unitaires
        normes = np.linalg.norm(normales, axis=1) #tableau avec les normes des normales
        normes_1= np.abs(normes - 1)
        normes_1_bool= normes_1 > tolerance #tableau de booléen si normes-1 > tolérance alors il y aura écrit True 
        non_unitaire = np.sum(normes_1_bool) #On compte les True (ils valent pour 1 et les False pour 0)
        erreure_non_unitaire = (non_unitaire / len(normales)) * 100

        # Vérification de la règle de la main droite
        calcul_normales = np.cross(triangles[:, 1] - triangles[:, 0], #on fait les produits vectoriels des vecteurs
                                       triangles[:, 2] - triangles[:, 0])
        print(calcul_normales)
        calcul_normales = calcul_normales/np.linalg.norm(calcul_normales, axis=1)[:,None] #on les normalises
        print(calcul_normales.shape,normales.shape)

        produits_scalaires = np.sum(normales * calcul_normales, axis=1) # '*' fait un produit terme à terme et np.sum permet finalement d'avoir le produit scalaire au carré, comme ce sont des vecteurs unitaires cela devrait donner 1 si les quantités sont identiques
        normales_non_identiques = np.sum(np.abs(produits_scalaires - 1) > tolerance) 
        erreure_normales_non_identiques = (normales_non_identiques / len(normales)) * 100
        return erreure_normales_non_identiques,  erreure_non_unitaire

    #Problème avec les tailles de 'normales' et 'calcul_normales'

    with open("data/cube.stl", mode="rt", encoding="us-ascii") as cube3_file:
        cube3_stl = cube3_file.read()
    tokens10 = tokenize(cube2_stl)
    triangles10, normales10, name10 = parse(tokens10)
    print (orientation_rule(triangles10,normales10))


    with open("data/teapot.stl", mode="rt", encoding="us-ascii") as tea3_file:
        tea3_stl = tea3_file.read()
    tokens6 = tokenize(tea3_stl)
    triangles6, normales6, name6 = parse(tokens6)
    print (orientation_rule(triangles6,normales6))


    return (
        cube3_file,
        cube3_stl,
        name10,
        name6,
        normales10,
        normales6,
        orientation_rule,
        tea3_file,
        tea3_stl,
        tokens10,
        tokens6,
        triangles10,
        triangles6,
    )


@app.cell
def __(mo):
    mo.md(
    rf"""
    ## OBJ Format

    The OBJ format is an alternative to the STL format that looks like this:

    ```
    # OBJ file format with ext .obj
    # vertex count = 2503
    # face count = 4968
    v -3.4101800e-003 1.3031957e-001 2.1754370e-002
    v -8.1719160e-002 1.5250145e-001 2.9656090e-002
    v -3.0543480e-002 1.2477885e-001 1.0983400e-003
    v -2.4901590e-002 1.1211138e-001 3.7560240e-002
    v -1.8405680e-002 1.7843055e-001 -2.4219580e-002
    ...
    f 2187 2188 2194
    f 2308 2315 2300
    f 2407 2375 2362
    f 2443 2420 2503
    f 2420 2411 2503
    ```

    This content is an excerpt from the `data/bunny.obj` file.

    """
    )
    return


@app.cell
def __(mo, show):
    mo.show_code(show("data/bunny.obj", scale="1.5"))
    return


@app.cell
def __(mo):
    mo.md(
        """
        Study the specification of the OBJ format (search for suitable sources online),

        then develop a `OBJ_to_STL` function that is rich enough to convert the OBJ bunny file into a STL bunny file.
        """
    )
    return


@app.cell
def __(np, re):
    def OBJ_to_STL(fichier_entree, stl_sortie):
        
        with open(fichier_entree, 'r', encoding='utf-8') as obj_file:
            sommets = []
            faces = []

            for line in obj_file:
                line = line.strip()

                if line.startswith('v '):
                    vertex = [float(x) for x in line.split()[1:4]]
                    sommets.append(vertex)

                elif line.startswith('f '):
                    parts = line.split()[1:]
                    face_indices = []
                    for part in parts:
                        index_str = re.split(r'/', part)[0]
                        face_indices.append(int(index_str) - 1)
                        
            sommets = np.array(sommets, dtype=np.float32)
            faces = np.array(faces, dtype=np.int32)

        with open(stl_sortie, 'w', encoding='utf-8') as stl_file:
            stl_file.write(f"solid {fichier_entree}\n")

            for face in faces:
                v1 = sommets[face[0]]
                v2 = sommets[face[1]]
                v3 = sommets[face[2]]

                normal = np.cross(v2 - v1, v3 - v1)
                normal = normal / np.linalg.norm(normal)

                stl_file.write(f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n")
                stl_file.write("    outer loop\n")
                stl_file.write(f"      vertex {v1[0]} {v1[1]} {v1[2]}\n")
                stl_file.write(f"      vertex {v2[0]} {v2[1]} {v2[2]}\n")
                stl_file.write(f"      vertex {v3[0]} {v3[1]} {v3[2]}\n")
                stl_file.write("    endloop\n")
                stl_file.write("  endfacet\n")

            stl_file.write(f"endsolid {fichier_entree}\n")
        print(f"Conversion terminée (sans gestion d'erreurs) : {fichier_entree} -> {stl_sortie}")

    OBJ_to_STL("data/bunny.obj","data/bunny.stl")
    return (OBJ_to_STL,)


@app.cell
def __(mo):
    mo.md(
        rf"""
    ## Binary STL

    Since the STL ASCII format can lead to very large files when there is a large number of facets, there is an alternate, binary version of the STL format which is more compact.

    Read about this variant online, then implement the function

    ```python
    def STL_binary_to_text(stl_filename_in, stl_filename_out):
        pass  # 🚧 TODO!
    ```

    that will convert a binary STL file to a ASCII STL file. Make sure that your function works with the binary `data/dragon.stl` file which is an example of STL binary format.

    💡 The `np.fromfile` function may come in handy.

        """
    )
    return


@app.cell
def __(mo, show):
    mo.show_code(show("data/dragon.stl", theta=75.0, phi=-20.0, scale=1.7))
    return


@app.cell
def __(make_STL, np):
    def STL_binary_to_text(stl_filename_in, stl_filename_out):
        with open(stl_filename_in, mode="rb") as file:
            _ = file.read(80)
            m = np.fromfile(file, dtype=np.uint32, count=1)[0]
            normals = []
            faces = []
            for i in range(m):
                normals.append(np.fromfile(file, dtype=np.float32, count=3))
                faces.append(np.fromfile(file, dtype=np.float32, count=9).reshape(3, 3))
                _ = file.read(2)
        faces = np.array(faces)
        normals=np.array(normals)
        stl_text = make_STL(faces, normals)
        with open(stl_filename_out, mode="wt", encoding="utf-8") as file:
            file.write(stl_text)


    STL_binary_to_text('data/dragon.stl','data/dragonascii.stl' )
    return (STL_binary_to_text,)


@app.cell
def __(mo):
    mo.md(rf"""## Constructive Solid Geometry (CSG)

    Have a look at the documentation of [{mo.icon("mdi:github")}fogleman/sdf](https://github.com/fogleman/) and study the basics. At the very least, make sure that you understand what the code below does:
    """)
    return


@app.cell
def __(X, Y, Z, box, cylinder, mo, show, sphere):
    demo_csg = sphere(1) & box(1.5)
    _c = cylinder(0.5)
    demo_csg = demo_csg - (_c.orient(X) | _c.orient(Y) | _c.orient(Z))
    demo_csg.save('output/demo-csg.stl', step=0.05)
    mo.show_code(show("output/demo-csg.stl", theta=45.0, phi=45.0, scale=1.0))
    return (demo_csg,)


@app.cell
def __(mo):
    mo.md("""ℹ️ **Remark.** The same result can be achieved in a more procedural style, with:""")
    return


@app.cell
def __(
    box,
    cylinder,
    difference,
    intersection,
    mo,
    orient,
    show,
    sphere,
    union,
):
    demo_csg_alt = difference(
        intersection(
            sphere(1),
            box(1.5),
        ),
        union(
            orient(cylinder(0.5), [1.0, 0.0, 0.0]),
            orient(cylinder(0.5), [0.0, 1.0, 0.0]),
            orient(cylinder(0.5), [0.0, 0.0, 1.0]),
        ),
    )
    demo_csg_alt.save("output/demo-csg-alt.stl", step=0.05)
    mo.show_code(show("output/demo-csg-alt.stl", theta=45.0, phi=45.0, scale=1.0))
    return (demo_csg_alt,)


@app.cell
def __():
    return


@app.cell
def __(mo):
    mo.md(
        rf"""
    ## JupyterCAD

    [JupyterCAD](https://github.com/jupytercad/JupyterCAD) is an extension of the Jupyter lab for 3D geometry modeling.

      - Use it to create a JCAD model that correspond closely to the `output/demo_csg` model;
    save it as `data/demo_jcad.jcad`.

      - Study the format used to represent JupyterCAD files (💡 you can explore the contents of the previous file, but you may need to create some simpler models to begin with).

      - When you are ready, create a `jcad_to_stl` function that understand enough of the JupyterCAD format to convert `"data/demo_jcad.jcad"` into some corresponding STL file.
    (💡 do not tesselate the JupyterCAD model by yourself, instead use the `sdf` library!)


        """
    )
    return


@app.cell
def __(box, json, sdf):
    # C'est le format JSON qui est utilisé pour retranscrire les fichiers JCAD

    #C'est comme un dictionnaire python où les clés sont toujours des str mais les valeurs peuvent être n'importe quoi




    def jcad_to_stl(fichier_entree,fichier_sortie):
        
        with open(fichier_entree, "r") as f:
            jcad_data = json.load(f)

        
        stock= sdf.box([0, 0, 0]) #Crée un cube d'un cube de côté 0


        for obj in jcad_data.get("objects"):

            if obj["name"] == "Box 1":
                size = (obj.get("Height"),obj.get("Length"),obj.get("Length"))
                Placement = obj.get("Placement")
              
                cube = sdf.box(size) # On ajoute le cube correspondant
                cube = cube.translate(Placement)
                stock = stock.union(box)

            elif obj["name"] == "Sphere 1":
                radius = obj.get("radius")
                Placement = obj.get("Placement")
               
                sphere = sdf.sphere(radius) #On ajoute la sphère correspondante
                sphere = sphere.translate(Placement)
                stock = stock.union(sphere)

            elif obj["name"] == "Cylinder 1":
                radius = obj.get("radius")
                height = obj.get("height")
                Placement = obj.get("Placement")
                
                cylinder = sdf.cylinder(radius) #On ajoute le cylindre correspondant
                cylinder = cylinder.translate(Placement)
                stock.union(cylinder)

            elif obj["name"] == "Cylinder 2":
                radius = obj.get("radius")
                height = obj.get("height")
                Placement = obj.get("Placement")
                
                cylinder = sdf.cylinder(radius) #On ajoute le cylindre correspondant
                cylinder = cylinder.translate(Placement)
                stock.union(cylinder)


            elif obj["name"] == "Cylinder 3":
                 radius = obj.get("radius")
                 height = obj.get("height")
                 Placement = obj.get("Placement")
                
                 cylinder = sdf.cylinder(radius) #On ajoute le cylindre correspondant
                 cylinder = cylinder.translate(Placement)
                 stock.union(cylinder)
            
            elif obj["name"] == "Cylinder 4":
                radius = obj.get("radius")
                height = obj.get("height")
                Placement = obj.get("Placement")
                
                cylinder = sdf.cylinder(radius) #On ajoute le cylindre correspondant
                cylinder = cylinder.translate(Placement)
                stock.union(cylinder)

            elif obj["name"] == "Cylinder 5":
                radius = obj.get("radius")
                height = obj.get("height")
                Placement = obj.get("Placement")
                
                cylinder = sdf.cylinder(radius) #On ajoute le cylindre correspondant
                cylinder = cylinder.translate(Placement)
                stock.union(cylinder)
       
        stock.save(fichier_sortie)


    jcad_to_stl("data/demo_jcad.jcad", "output/demo_model.stl")
    return (jcad_to_stl,)


@app.cell
def __(mo):
    mo.md("""## Appendix""")
    return


@app.cell
def __(mo):
    mo.md("""### Dependencies""")
    return


@app.cell
def __():
    # Python Standard Library
    import json

    # Marimo
    import marimo as mo

    # Third-Party Librairies
    import numpy as np
    import matplotlib.pyplot as plt
    import mpl3d
    from mpl3d import glm
    from mpl3d.mesh import Mesh
    from mpl3d.camera import Camera

    import meshio

    np.seterr(over="ignore")  # 🩹 deal with a meshio false warning

    import sdf
    from sdf import sphere, box, cylinder
    from sdf import X, Y, Z
    from sdf import intersection, union, orient, difference

    mo.show_code()
    return (
        Camera,
        Mesh,
        X,
        Y,
        Z,
        box,
        cylinder,
        difference,
        glm,
        intersection,
        json,
        meshio,
        mo,
        mpl3d,
        np,
        orient,
        plt,
        sdf,
        sphere,
        union,
    )


@app.cell
def __(mo):
    mo.md(r"""### STL Viewer""")
    return


@app.cell
def __(Camera, Mesh, glm, meshio, mo, plt):
    def show(
        filename,
        theta=0.0,
        phi=0.0,
        scale=1.0,
        colormap="viridis",
        edgecolors=(0, 0, 0, 0.25),
        figsize=(6, 6),
    ):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1], xlim=[-1, +1], ylim=[-1, +1], aspect=1)
        ax.axis("off")
        camera = Camera("ortho", theta=theta, phi=phi, scale=scale)
        mesh = meshio.read(filename)
        sommets = glm.fit_unit_cube(mesh.points)
        faces = mesh.cells[0].data
        sommets = glm.fit_unit_cube(sommets)
        mesh = Mesh(
            ax,
            camera.transform,
            sommets,
            faces,
            cmap=plt.get_cmap(colormap),
            edgecolors=edgecolors,
        )
        return mo.center(fig)

    mo.show_code()
    return (show,)


@app.cell
def __(mo, show):
    mo.show_code(show("data/teapot.stl", theta=45.0, phi=30.0, scale=2))
    return


if __name__ == "__main__":
    app.run()
