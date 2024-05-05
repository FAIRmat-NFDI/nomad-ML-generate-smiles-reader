#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np

from nomad.metainfo import Quantity, Package
from nomad.metainfo.metainfo import _placeholder_quantity
from nomad.datamodel.data import ArchiveSection


m_package = Package()


class PhysicalProperty(ArchiveSection):
    """
    A base section used to define the physical properties obtained in a simulation, experiment, or in a post-processing
    analysis. The main quantity of the `PhysicalProperty` is `value`, whose instantiation has to be overwritten in the derived classes
    when inheriting from `PhysicalProperty`. It also contains `rank`, to define the tensor rank of the physical property, and
    `variables`, to define the variables over which the physical property varies (see variables.py). This class can also store several
    string identifiers and quantities for referencing and establishing the character of a physical property.
    """

    # * `value` must be overwritten in the derived classes defining its type, unit, and description
    value: Quantity = _placeholder_quantity


class TotalEnergy(PhysicalProperty):
    """ """

    value = Quantity(
        type=np.float64,
        shape=[],
        unit='kcal/mol',  # confirmed units
        description="""
        Total energy from semiempirical VAMP calculation. This value is dependent on the basis selected
        and should not be used as an absolute value. Difference between Electronic Energy and Repulsive Energy.
        """,
    )


class ElectronicEnergy(PhysicalProperty):
    """ """

    value = Quantity(
        type=np.float64,
        shape=[],
        unit='kcal/mol',  # confirmed units
        description="""
        Electron energy from semiempirical VAMP calculation. This value is dependent on the basis selected.
        """,
    )


class RepulsiveEnergy(PhysicalProperty):
    """ """

    value = Quantity(
        type=np.float64,
        shape=[],
        unit='kcal/mol',  # confirmed units
        description="""
        Repulsive Energy from semiempirical VAMP calculation. This is the core-core repulsion energy. This value
        is depednent on the basis selected.
        """,
    )


class IonizationPotential(PhysicalProperty):
    """ """

    value = Quantity(
        type=np.float64,
        shape=[],
        unit='kcal/mol',  # confirmed units
        description="""
        Ionization Potential from semiempirical VAMP calculation.
        """,
    )


class GapEnergy(PhysicalProperty):
    """ """

    value_homo = Quantity(
        type=np.float64,
        shape=[],
        unit='eV',  # confirmed units
        description="""
        Highest occupied molecular orbital value.
        """,
    )

    value_lumo = Quantity(
        type=np.float64,
        shape=[],
        unit='eV',  # confirmed units
        description="""
        Lowest unoccupied molecular orbital value.
        """,
    )

    value = Quantity(
        type=np.float64,
        shape=[],
        unit='eV',  # confirmed units
        description="""
        Value of the gap of energies. This is calculated as the difference `value_homo - value_lumo`; if the
        `value` is negative, we don't set it.
        """,
    )

    def extract_gap(self) -> None:
        if self.value_homo and self.value_lumo:
            value = self.value_homo - self.value_lumo
            if value.magnitude > 0.0:
                self.value = value


class HeatOfFormation(PhysicalProperty):
    """ """

    value = Quantity(
        type=np.float64,
        shape=[],
        unit='kcal/mol',  # confirmed units
        description="""
        Heat of Formation. A fundamental inconsistency inherent in the parameterization of semiempirical methods to
        reproduce heats of formation. The energy calculated by VAMP is the internal energy of a hypothetical
        motionless (Born-Oppenheimer) state. To relate this energy to heats of formation at 298 K, an atom-based
        scheme is used, assuming that the energy difference between the Born-Oppenheimer state and the molecule at
        298 K can be treated in an additive fashion. JANAF Thermochemical Tables 2nd Edition are used for molecule.
        This means that the zero-point energy and the energy required to warm the molecule to 298 K are assumed to
        be identical for isomers. This is clearly not the case and can lead to errors of up to 5 kcal mol-1.
        """,  #
    )


class MultipoleMoment(PhysicalProperty):
    """ """

    value = Quantity(
        type=np.float64,
        shape=[],
        unit='debye',  # confirmed units
        description="""
        Value of the total dipole moment of the system.
        """,
    )

    value_dipole = Quantity(
        type=np.float64,
        shape=[3],
        unit='debye',  # confirmed units
        description="""
        Value of the X,Y,Z component of the dipole moment vector of the system.
        """,
    )

    # value_quadrupole = Quantity(
    #     type=np.float64,
    #     shape=[3, 3],
    #     unit='dimensionless',
    #     description="""
    #     Value of the dipole moment.
    #     """,
    # )

    # value_octupole = Quantity(
    #     type=np.float64,
    #     shape=[3, 3, 3, 3],
    #     unit='dimensionless',
    #     description="""
    #     Value of the octupole moment.
    #     """,
    # )


class Enthalpy(PhysicalProperty):
    """ """

    value = Quantity(
        type=np.float64,
        shape=[],
        unit='kcal/mol',  # confirmed units
        description="""
        Enthalpy at 298K. Thermodynamics calculations yield enthalpy changes between the Born-Oppenheimer
        state and the temperature in question. A fictitious Born-Oppenheimer energy is used in thermodynamics calculations.
        The best procedure is simply to ignore the "heat of formation" and to calculate thermodynamic quantities for
        reactions based on the Born-Oppenheimer energy and the calculated enthalpy changes, as for ab initio calculations.
        The electronic contribution is determined based on ideal gas apprximation. The vibrational contribution is determined
        based on vibrational calculation and equation found in Hirano et al.,1993.
        """,
    )


class Entropy(PhysicalProperty):
    """ """

    value = Quantity(
        type=np.float64,
        shape=[],
        unit='cal/K/mol',  # confirmed units
        description="""
        Entropy at 298K.Thermodynamics calculations yield entropy changes between the Born-Oppenheimer
        state and the temperature in question. A fictitious Born-Oppenheimer energy is used in thermodynamics calculations.
        The best procedure is simply to ignore the "heat of formation" and to calculate thermodynamic quantities for
        reactions based on the Born-Oppenheimer energy and the calculated entropy changes, as for ab initio calculations.
        The vibrational contribution is determined based on vibrational calculation and equation found in Hirano et al.,1993.
        """,
    )


class HeatCapacity(PhysicalProperty):
    """ """

    value = Quantity(
        type=np.float64,
        shape=[],
        unit='cal/K/mol',  # confirmed units
        description="""
        Heat capacity at 298K for both Electronic and Vibrational Contribution. The heat capacity is calculated at constant
        pressure, Cp, based on the ideal gas for the translational and rotational terms. Vibrational contribution is calculated
        based on the vibrational calculation and equation found in Hirano et al.,1993.
        """,
    )


class ZeroPointEnergy(PhysicalProperty):
    """ """

    value = Quantity(
        type=np.float64,
        shape=[],
        unit='kcal/mol',  # confirmed units
        description="""
        Zero point energy.
        """,
    )


class ElectronicLevels(PhysicalProperty):
    """ """

    type = Quantity(
        type=str,
        shape=[],
        description="""
        Type of electronic levels. For example, UV-VIS.
        """,
    )

    n_levels = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Number of electronic levels.
        """,
    )

    excited_state = Quantity(
        type=np.int32,
        shape=['n_levels'],
        description="""
        Excited state number. Listed from lowest to highest energy levels.
        """,
    )

    value = Quantity(
        type=np.float64,
        shape=['n_levels'],
        unit='eV',  # confirmed units
        description="""
        Electronic transition energy.
        """,
    )

    value_wavelength = Quantity(
        type=np.float64,
        shape=['n_levels'],
        unit='nm',  # confirmed units
        description="""
        Electronic transition in wavelength.
        """,
    )

    oscillator_strength = Quantity(
        type=np.float64,
        shape=['n_levels'],
        unit='dimensionless',  # confirmed units
        description="""
        Electronic transition strength. Determined based on overlap of the initial and final wavefunctions.
        """,
    )

    transition_type = Quantity(
        type=np.int32,
        shape=['n_levels'],
        unit='dimensionless',  # confirmed units
        description="""
        Transition type of the electronic level. For example, 1 for singlet and 3 for triplet. Singlet state
        is a state with all electron spins paired. Triplet state is a state with two unpaired electrons.
        """,
    )


class VibrationalModes(PhysicalProperty):
    """ """

    n_modes = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Number of vibrational or phonon mode number. Ascending order based on inverse wavelength or frequency.
        """,
    )

    value = Quantity(
        type=str,
        shape=['n_modes'],
        description="""
        Vibrational modes type. For example, A=Acoustic and O=Optical.
        """,
    )

    frequency = Quantity(
        type=np.float64,
        shape=['n_modes'],
        unit='1/cm',  # confirmed units
        description="""
        Vibrational modes inverse wavelength, which is proportional to the frequency.
        """,
    )

    reduced_mass = Quantity(
        type=np.float64,
        shape=['n_modes'],
        unit='dimensionless',  # confirmed units
        description="""
        Vibrational modes reduced masses.
        """,
    )

    raman_intensity = Quantity(
        type=np.float64,
        shape=['n_modes'],
        unit='dimensionless',  # confirmed units
        description="""
        Vibrational mode Raman intensities.
        """,
    )


class VibrationalSpectrum(PhysicalProperty):
    """ """

    n_frequencies = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Number of frequencies.
        """,
    )

    frequency = Quantity(
        type=np.float64,
        shape=['n_frequencies'],
        unit='1/cm',  # confirmed units
        description="""
        Vibrational modes inverse wavelength, which is proportional to the frequency.
        """,
    )

    value = Quantity(
        type=np.float64,
        shape=['n_frequencies'],
        unit='meter / mol',
        description="""
        Vibrational spectrum intensity values.
        """,
    )


m_package.__init_metainfo__()
