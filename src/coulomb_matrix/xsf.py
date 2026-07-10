import numpy as np

"""
Simple XSF reader adapted from existing xsf_2_np.py
"""


class Xsf2Np:
    def __init__(self, file_obj, skip_wf=False):
        """Initialize the Xsf2Np object.

        Parameters
        ----------
        file_obj : file-like object
            A file-like object containing the XSF data.
        skip_wf : bool, optional
            If True, skip reading the Wannier function data. This enables faster reading when only lattice information is needed. Default is False.
        """
        self.file_obj = file_obj
        self.skip_wf = skip_wf
        (
            self.lattice_vectors,
            self.atoms,
            self.nx,
            self.ny,
            self.nz,
            self.WF,
            self.origin,
            self.supercell,
        ) = self.read_xsf()

    def read_xsf(self):
        for line in self.file_obj:
            line = self.linereader(line)
            if line == "PRIMVEC":
                lattice_vectors = np.zeros((3, 3))
                for i in range(3):
                    line = self.file_obj.readline()
                    lattice_vectors[i] = np.array([float(x) for x in line.split()])
            elif line == "CONVVEC":
                for i in range(3):
                    line = self.file_obj.readline()
            elif line == "PRIMCOORD":
                line = self.file_obj.readline()
                num_atoms = int(line.split()[0])
                atoms = []
                for a in range(num_atoms):
                    line = self.file_obj.readline()
                    type, ax, ay, az = line.split()
                    ax, ay, az = float(ax), float(ay), float(az)
                    atoms.append({type: [ax, ay, az]})
            elif line == "BEGIN_DATAGRID_3D_UNKNOWN":
                line = self.file_obj.readline()
                nx, ny, nz = line.split()
                nx, ny, nz = int(nx), int(ny), int(nz)
                line = self.file_obj.readline()
                origin = np.array([float(x) for x in line.split()])
                supercell = np.zeros((3, 3))
                itr = 0
                for i in range(3):
                    line = self.file_obj.readline()
                    supercell[i] = np.array([float(x) for x in line.split()])
                WF = None
                if not self.skip_wf:
                    WF = np.zeros((nx, ny, nz))
                    for iz in range(nz):
                        for iy in range(ny):
                            for ix in range(nx):
                                tr = iz * ny * nx + iy * nx + ix
                                if not tr % 6:
                                    itr += 1
                                    line = self.file_obj.readline()
                                    values = [float(x) for x in line.split()]
                                WF[ix, iy, iz] = values[tr % 6]
            elif line == "END_DATAGRID_3D":
                return lattice_vectors, atoms, nx, ny, nz, WF, origin, supercell

    def get_lattice_vectors(self):
        return self.lattice_vectors

    def get_atoms(self):
        return self.atoms

    def get_origin(self):
        return self.origin

    def get_real_space_grid(self):
        return self.nx, self.ny, self.nz

    def get_supercell(self):
        return self.supercell

    def get_WF(self):
        return self.WF

    def linereader(self, line):
        line = line.strip()
        if not line or line.startswith("#"):
            self.linereader(self.file_obj.readline())
        return line


if __name__ == "__main__":
    path = "/usr/scratch/bucaramanga/awinka/MoS2/proj/wannier90_00001.xsf"
    with open(path, "r") as file:
        xsf = Xsf2Np(file)
