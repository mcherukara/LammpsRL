from imports import *
import params

class lammpsrun:
    def __init__(self, cores=1):
        self.lmp = LammpsLibrary(cores=cores)
        self.lmp.file(params.infile) #Read lammps file with everything except minimize
        self.deleted_atoms = [] #List of atoms deleted
        self.orig_ats = self.get_atoms()
        self.orig_topSats = self.orig_ats[np.where(self.orig_ats[:,1]==params.atom_type)]
        self.debug = 1
    
    def reset (self):
        self.lmp.command("clear")
        self.lmp.file(params.infile)
        self.deleted_atoms = [] #List of atoms deleted
        self.orig_ats = self.get_atoms()
        self.orig_topSats = self.orig_ats[np.where(self.orig_ats[:,1]==params.atom_type)]
        
    def get_pe_atom (self):
        self.lmp.command("run 0 pre no post no")
        pe = self.lmp.pe
        nats = self.lmp.natoms
        #kcal_mol_to_ev_atom = 4.3363*10**-2
        pe_atom = pe/nats#*kcal_mol_to_ev_atom
        if self.debug:
            print ("Atoms", nats)
            print ("PE/atom", pe_atom)
        return pe_atom
    
    #Get atom position, id and type
    def get_atoms (self):
        x = self.lmp.gather_atoms("x")
        types = self.lmp.gather_atoms("type")
        ids = self.lmp.gather_atoms("id")
        #Stack them in array
        ats = np.vstack([ids,types,x[:,0],x[:,1],x[:,2]])
        ats = ats.transpose()
        return ats
    
    def delete_atoms (self, ats_to_delete):
        if self.debug:
            print(self.deleted_atoms)

        #Convert atom ids list to string
        strlist = map(str, ats_to_delete)
        strlist = ' '.join(strlist)

        #Group atoms to be deleted by id and then delete
        grpcmd = "group delS id " + strlist
        self.lmp.command(grpcmd)
        self.lmp.command("delete_atoms group delS")
        
    def viz_atoms (self, shape): #Encodes one-hot of vacancies in shape*shape
        viz_array = np.zeros((shape,shape),int)
        
        #Get current box bounds
        xmin = self.lmp.extract_box()[0][0]
        xmax = self.lmp.extract_box()[1][0]
        ymin = self.lmp.extract_box()[0][1]
        ymax = self.lmp.extract_box()[1][1]
        
        delx = (xmax-xmin)/shape
        dely = (ymax-ymin)/shape
        
        for atom in self.deleted_atoms:
            loc = np.where(self.orig_topSats[:,0]==atom)[0][0]
            #Find location in array of deleted atom (in case scrambled)
            x,y = self.orig_topSats[loc,2], self.orig_topSats[loc,3]
            # Get x, y in Angstroms
            ax = int(m.floor(((x-xmin)/delx)))
            ay = int(m.floor(((y-ymin)/dely)))
            viz_array[ax, ay] = 1 #Set closest grid point to 1
        
        return viz_array