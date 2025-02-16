"""
Dihedral angle analyser working on NMRlipids Databank.

TODO: add usage example
TODO: test it with the newest Databank version
"""

import copy
import os

import DatabankLib
from urllib import request

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.lib.distances import calc_dihedrals

from DatabankLib.core import initialize_databank
from DatabankLib.databankio import resolve_download_file_url
from DatabankLib.databankLibrary import loadMappingFile


def make_positive_angles(x):
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = x[i] + 360
        else:
            x[i] = x[i]
    return x

class DihedralFromAtoms(AnalysisBase):
    """Calculate dihedral angles for specified atom groups.

    Dihedral angles will be calculated for each atom group that is given for
    each step in the trajectory. Each :class:`~MDAnalysis.core.groups.AtomGroup`
    must contain 4 atoms.

    Note
    ----
    This class takes a list as an input and is most useful for a large
    selection of atom_groups. If there is only one atom_group of interest, then
    it must be given as a list of one atom_group.

    """

    def __init__(self, atom_groups, orders, **kwargs):
        """Parameters
        ----------
        atom_groups : Iterable
                        a list of atom groups for which the dihedral angles are calculated

        Raises
        ------
        ValueError
                        If any atom groups do not contain 4 atoms

        """
        super(DihedralFromAtoms, self).__init__(
            atom_groups[0].universe.trajectory, **kwargs
        )
        self.atom_groups = atom_groups

        if any([len(ag) != 4 for ag in atom_groups]):
            raise ValueError("All AtomGroups must contain 4 atoms")

        if len(atom_groups) != len(orders):
            raise ValueError("Order data should be provided for every atom group")

        self.ag1 = mda.AtomGroup(
            [atom_groups[i][orders[i][0]] for i in range(len(atom_groups))]
        )
        self.ag2 = mda.AtomGroup(
            [atom_groups[i][orders[i][1]] for i in range(len(atom_groups))]
        )
        self.ag3 = mda.AtomGroup(
            [atom_groups[i][orders[i][2]] for i in range(len(atom_groups))]
        )
        self.ag4 = mda.AtomGroup(
            [atom_groups[i][orders[i][3]] for i in range(len(atom_groups))]
        )

    def _prepare(self):
        self.angles = []

    def _single_frame(self):
        angle = calc_dihedrals(
            self.ag1.positions,
            self.ag2.positions,
            self.ag3.positions,
            self.ag4.positions,
            box=self.ag1.dimensions,
        )
        self.angles.append(angle)

    def _conclude(self):
        self.angles = np.rad2deg(np.array(self.angles))


def process_lipid_dihedrals(lipids, DIHatoms):
    colors = {"POPC": "black", "POPS": "red", "POPE": "blue", "POPG": "green"}
    systems = initialize_databank()

    for readme in systems:
        local_path: str = os.path.join(DatabankLib.NMLDB_SIMU_PATH, readme["path"])
        for molname in lipids:
            doi = readme.get("DOI")
            trj = readme.get("TRJ")
            tpr = readme.get("TPR")

            trj_name = os.path.join(local_path, readme.get("TRJ")[0][0])
            tpr_name = os.path.join(local_path, readme.get("TPR")[0][0])
            gro_name = os.path.join(local_path, "conf.gro")
            trj_url = resolve_download_file_url(doi, trj[0][0])
            tpr_url = resolve_download_file_url(doi, tpr[0][0])

            # Download tpr and xtc files to same directory where dictionary and data are located
            if not os.path.isfile(tpr_name):
                response = request.urlretrieve(tpr_url, tpr_name)

            if not os.path.isfile(trj_name):
                response = request.urlretrieve(trj_url, trj_name)

            if sum(readme["N" + molname]) > 0:
                print("Analyzing " + molname + " in " + readme.get("path"))
                # fig= plt.figure(figsize=(12,9))
                if not os.path.isfile(gro_name):
                    os.system(
                        "echo System | gmx trjconv -f {} -s {}  -dump 0 -o {}".format(
                            trj_name, tpr_name, gro_name
                        )
                    )

                xtc_whole = os.path.join(local_path, "/whole.xtc")
                if not os.path.isfile(xtc_whole):
                    os.system(
                        "echo System | gmx trjconv -f {} -s {} -o {} -pbc mol ".format(
                            trj_name, tpr_name, xtc_whole
                        )
                    )

                try:
                    traj = mda.Universe(tpr_name, xtc_whole)
                except FileNotFoundError or OSError:
                    continue

                mapping_dict = loadMappingFile(readme["MAPPING_DICT"][molname])
                try:
                    atom1 = mapping_dict[DIHatoms[0]]
                    atom2 = mapping_dict[DIHatoms[1]]
                    atom3 = mapping_dict[DIHatoms[2]]
                    atom4 = mapping_dict[DIHatoms[3]]
                    print(atom1, atom2, atom3, atom4)
                except:
                    print("Some atom not found in the mapping file.")
                    continue

                ags = []
                orders = []
                for residue in traj.select_atoms(
                    "name {} {} {} {}".format(atom1, atom2, atom3, atom4)
                ).residues:
                    atoms = traj.select_atoms(
                        "name {} {} {} {} and resid {}".format(
                            atom1, atom2, atom3, atom4, str(residue.resid)
                        )
                    )
                    if len(atoms) == 4:
                        # print(atoms.names)
                        ags.append([])
                        ags[-1] = copy.deepcopy(atoms)
                        orders.append(
                            [
                                np.where(atoms.names == atom1)[0][0],
                                np.where(atoms.names == atom2)[0][0],
                                np.where(atoms.names == atom3)[0][0],
                                np.where(atoms.names == atom4)[0][0],
                            ]
                        )
                R = DihedralFromAtoms(ags, orders).run()
                dihRESULT = R.angles.T

                dihRESULT = [make_positive_angles(x) for x in dihRESULT]
                distSUM = np.zeros(360)
                for i in dihRESULT:
                    distSUM += np.histogram(
                        i, np.arange(0, 361, 1), density=True
                    )[0]

                distSUM = [x / len(dihRESULT) for x in distSUM]
                xaxis = (
                    np.histogram(i, np.arange(0, 361, 1), density=True)[1][
                        1:
                    ]
                    + np.histogram(i, np.arange(0, 361, 1), density=True)[
                        1
                    ][:-1]
                ) / 2.0

                dihedral_folders = local_path.replace("Simulations", "dihedral")
                os.system(f"mkdir -p {dihedral_folders}")
                os.system(
                    f"cp {os.path.join(local_path,readme.get('path'),'README.yaml')} {dihedral_folders}"
                )
                outfile_name = os.path.join(dihedral_folders, f"{molname}_{'_'.join(DIHatoms)}.dat")
                with open(outfile_name, "w") as outfile:
                    for i in range(len(xaxis)):
                        outfile.write(
                            str(xaxis[i]) + " " + str(distSUM[i]) + "\n"
                        )

