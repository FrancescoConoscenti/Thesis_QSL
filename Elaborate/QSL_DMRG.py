import numpy as np
import matplotlib.pyplot as plt
import os

# TeNPy imports
from tenpy.networks.site import SpinSite
from tenpy.networks.mps import MPS
from tenpy.models.model import CouplingMPOModel
from tenpy.models.lattice import Square
from tenpy.algorithms import dmrg


def calculate_R(S_q, L):
    """
    Calculates the peak sharpness ratio R = 1 - S(Q+dq)/S(Q).
    Works with S_q being the 2D FFT of the correlation function on an LxL grid.
    """
    S_q_copy = np.abs(S_q.copy())
    # Avoid selecting the trivial q=(0,0) (total structure factor) as the peak
    S_q_copy[0, 0] = 0.0
    peak_idx = np.unravel_index(np.argmax(S_q_copy), S_q.shape)
    Q_val = S_q_copy[peak_idx]

    if Q_val == 0:
        return 0.0

    # neighbors in k-space (wrap-around)
    neighbor_x_idx = ((peak_idx[0] + 1) % L, peak_idx[1])
    neighbor_y_idx = (peak_idx[0], (peak_idx[1] + 1) % L)

    S_Q_plus_dq_x = S_q_copy[neighbor_x_idx]
    S_Q_plus_dq_y = S_q_copy[neighbor_y_idx]

    R_x = 1.0 - (S_Q_plus_dq_x / Q_val)
    R_y = 1.0 - (S_Q_plus_dq_y / Q_val)
    R_avg = 0.5 * (R_x + R_y)
    return R_avg


class J1J2Heisenberg(CouplingMPOModel):
    """A TeNPy model for the J1-J2 Heisenberg model on a square lattice."""
    def __init__(self, model_params):
        # Let CouplingMPOModel handle lattice initialization from model_params.
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        return SpinSite(S=0.5, conserve='Sz')

    def init_terms(self, model_params):
        J1 = model_params.get('J1', 1.0)
        J2 = model_params.get('J2', 0.0)

        # nearest neighbors (J1)
        for u1, u2, dx in self.lat.pairs.get('nearest_neighbors', []):
            self.add_coupling(J1, u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            self.add_coupling(2.0 * J1, u1, 'Sz', u2, 'Sz', dx)

        # next-nearest neighbors (J2)
        for u1, u2, dx in self.lat.pairs.get('next_nearest_neighbors', []):
            self.add_coupling(J2, u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            self.add_coupling(2.0 * J2, u1, 'Sz', u2, 'Sz', dx)


def main():
    lattice_sizes = [5]  # use even sizes for simple Sz=0 Neel initial state
    J_values = np.linspace(0.4, 0.7, 10)

    plt.figure(figsize=(8, 6))

    for L in lattice_sizes:
        R_values = []
        print(f"\nStarting DMRG analysis for L={L}x{L} and J2 values: {J_values}")

        for J2 in J_values:
            print(f"\n--- Processing L={L}, J2 = {J2:.3f} ---")

            # 1) Create the Spin site and Square lattice (note: 'site=' keyword)
            site = SpinSite(S=0.5, conserve='Sz')
            lat = Square(Lx=L, Ly=L, site=site, bc='periodic', bc_MPS='finite')

            # 2) Model params: pass the lattice under the 'lattice' key
            model_params = {
                'J1': 1.0,
                'J2': J2,
                'lattice': lat,
                'verbose': 0
            }
            M = J1J2Heisenberg(model_params)

            # -----------------------
            # 3) Robust product_state -> MPS construction
            # -----------------------

            # get mps_sites (some TenPy versions provide method, others attribute)
            try:
                mps_sites = lat.mps_sites()
            except TypeError:
                # attribute callable mismatch
                try:
                    mps_sites = lat.mps_sites
                except Exception:
                    # fallback: repeat single site object
                    print("Warning: lat.mps_sites() not available; falling back to repeating site.")
                    mps_sites = [site] * lat.N_sites
            except Exception:
                # other failure fallback
                mps_sites = [site] * lat.N_sites

            N_mps_sites = len(mps_sites)
            print(f"Debug: lat.N_sites = {lat.N_sites}, len(mps_sites) = {N_mps_sites}")

            # helper to map mps index -> (x, y) coordinates (robust across TenPy versions)
            def idx_to_xy(lat_obj, idx):
                if hasattr(lat_obj, "mps2lat_idx"):
                    arr = lat_obj.mps2lat_idx(idx)
                    return int(arr[0]), int(arr[1])
                if hasattr(lat_obj, "i_to_x") and hasattr(lat_obj, "i_to_y"):
                    return int(lat_obj.i_to_x[idx]), int(lat_obj.i_to_y[idx])
                if hasattr(lat_obj, "lat_idx"):
                    arr = lat_obj.lat_idx(idx)
                    return int(arr[0]), int(arr[1])
                # last resort: assume row-major ordering on LxL
                x = idx % L
                y = idx // L
                return x, y

            # build a Neel product state in the exact MPS ordering
            product_state = []
            for i_mps in range(N_mps_sites):
                x, y = idx_to_xy(lat, i_mps)
                product_state.append("up" if (x + y) % 2 == 0 else "down")

            if len(product_state) != N_mps_sites:
                raise RuntimeError(
                    f"product_state length ({len(product_state)}) != number of MPS sites ({N_mps_sites})"
                )

            # Try constructing MPS using preferred API(s)
            psi = None
            # Option A: MPS.from_product_state
            try:
                psi = MPS.from_product_state(mps_sites, product_state, bc='finite')
                print("Constructed MPS with MPS.from_product_state(...).")
            except Exception as e_mps:
                # Option B: model helper
                try:
                    psi = M.psi_from_product_state(product_state)
                    print("Constructed MPS with M.psi_from_product_state(...).")
                except Exception as e_model:
                    # If both fail, give clear diagnostics
                    raise RuntimeError(
                        "Failed to construct MPS from product_state. "
                        f"MPS.from_product_state error: {repr(e_mps)} ; M.psi_from_product_state error: {repr(e_model)}"
                    )

            # -----------------------
            # 4) DMRG run
            # -----------------------
            dmrg_params = {
                'mixer': None,
                'max_E_err': 1.e-8,
                'trunc_params': {'chi_max': 1000, 'svd_min': 1.e-8},
                'max_sweeps': 10,
                'verbose': 0,
            }
            results = dmrg.run(psi, M, dmrg_params)
            E = results.get('E', None)
            if E is not None:
                print(f"DMRG complete. Ground state energy: {E:.6f}")
            else:
                print("DMRG complete. (no energy returned in results dict)")

            # -----------------------
            # 5) Correlations -> structure factor -> R
            # -----------------------
            Sz_Sz = psi.correlation_function("Sz", "Sz")
            Sp_Sm = psi.correlation_function("Sp", "Sm")

            # Prepare arrays for accumulated correlations and counts
            corr_r = np.zeros((L, L), dtype=np.complex128)
            counts = np.zeros((L, L), dtype=int)

            # robust index -> (x,y,u) helper for accumulation
            def idx_to_xyz(lat_obj, idx):
                if hasattr(lat_obj, "mps2lat_idx"):
                    a = lat_obj.mps2lat_idx(idx)
                    # ensure 3 entries (x,y,u)
                    if len(a) >= 3:
                        return int(a[0]), int(a[1]), int(a[2])
                    return int(a[0]), int(a[1]), 0
                if hasattr(lat_obj, "i_to_x") and hasattr(lat_obj, "i_to_y"):
                    return int(lat_obj.i_to_x[idx]), int(lat_obj.i_to_y[idx]), 0
                if hasattr(lat_obj, "lat_idx"):
                    a = lat_obj.lat_idx(idx)
                    if len(a) >= 3:
                        return int(a[0]), int(a[1]), int(a[2])
                    return int(a[0]), int(a[1]), 0
                # fallback
                x = idx % L
                y = idx // L
                return x, y, 0

            N_sites = lat.N_sites
            for i_site in range(N_sites):
                xi, yi, _ = idx_to_xyz(lat, i_site)
                for j_site in range(N_sites):
                    xj, yj, _ = idx_to_xyz(lat, j_site)
                    dx = (xi - xj) % L
                    dy = (yi - yj) % L
                    corr_r[dx, dy] += Sz_Sz[i_site, j_site] + 0.5 * Sp_Sm[i_site, j_site]
                    counts[dx, dy] += 1

            # average and guard against zero counts
            counts[counts == 0] = 1
            corr_r /= counts

            S_q = np.fft.fft2(corr_r)
            R = calculate_R(S_q, L)
            R_values.append(R)
            print(f"Calculated R = {R:.6f}")

        plt.plot(J_values, R_values, marker='o', linestyle='-', label=f'L={L}')

    plt.xlabel("$J_2$", fontsize=14)
    plt.ylabel("Peak sharpness R = 1 - S(Q+dq)/S(Q)", fontsize=14)
    plt.title("Structure Factor Peak Sharpness vs. $J_2$ (DMRG)", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    output_dir = "/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "QSL_regime_DMRG.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Analysis complete. Plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
