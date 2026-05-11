import math
import threading
import time
import os
import cv2
import numpy as np
from random import randrange, random, seed

from ga_chromosome import GAChromosome
from ga_constants import GAConstants

seed()


class GAEngine:
    """
    Genetic Algorithm engine to optimize calibration images and flags with CustomPattern
    """

    def __init__(self,
                 obj_points: list,
                 matched_points: list,
                 img_width: int,
                 img_height: int,
                 out_dir: str = "out",
                 progress_callback=None):
        self.obj_points       = obj_points
        self.matched_points   = matched_points
        self.img_width        = img_width
        self.img_height       = img_height
        self.out_dir          = out_dir
        self.progress_callback = progress_callback
        self.n_images         = len(obj_points)

        os.makedirs(out_dir, exist_ok=True)

        self._repo_thread  = {}
        self._repo_fitness = {}
        self._stop         = False

    # ------------------------------------------------------------------
    # Fitness: calibrate on subset and return reprojection error
    # ------------------------------------------------------------------
    def _fitness(self, chromosome: GAChromosome) -> float:
        indices = chromosome.active_indices()
        obj_pts = [self.obj_points[i] for i in indices]
        img_pts = [self.matched_points[i] for i in indices]

        flags = chromosome.get_flags()
        if flags & cv2.CALIB_USE_INTRINSIC_GUESS:
            f_guess = max(self.img_width, self.img_height)
            K_init = np.array([[f_guess, 0, self.img_width / 2.0],
                            [0, f_guess, self.img_height / 2.0],
                            [0, 0, 1.0]], dtype=np.float64)
            dist_init = np.zeros((5, 1), dtype=np.float64)
        else:
            K_init = None
            dist_init = None

        try:
            rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                obj_pts, img_pts,
                (self.img_width, self.img_height),
                K_init, dist_init,
                flags=flags
            )
        except cv2.error:
            return GAConstants.INF

        total_err = 0
        total_pts = 0
        for i in range(len(obj_pts)):
            proj, _ = cv2.projectPoints(
                np.array(obj_pts[i]), rvecs[i], tvecs[i], K, dist
            )
            a = np.array(img_pts[i], dtype=np.float32).reshape(-1, 1, 2)
            b = proj.reshape(-1, 1, 2).astype(np.float32)
            err = cv2.norm(a, b, cv2.NORM_L2)
            total_err += err ** 2
            total_pts += len(obj_pts[i])

        reproj = math.sqrt(total_err / total_pts) if total_pts > 0 else GAConstants.INF

        # Save intermediate result as XML
        key = GAChromosome.to_string(chromosome)
        fs_path = os.path.join(self.out_dir, key[:32] + '.xml')  # <-- xml
        fs = cv2.FileStorage(fs_path, cv2.FILE_STORAGE_WRITE)
        fs.write('cameraMatrix', K)
        fs.write('distCoeffs', dist)
        fs.write('ERR_REPROJ', reproj)
        fs.write('ERR_RMS', rms)
        fs.write('N_IMAGES', len(indices))
        fs.write('ACTIVE_INDICES', np.array(indices, dtype=int))
        fs.release()

        return reproj

    def _worker_fitness(self, chromosome: GAChromosome):
        key = GAChromosome.to_string(chromosome)
        self._repo_fitness[key] = (chromosome, self._fitness(chromosome))

    def _enqueue(self, chromosome: GAChromosome):
        key = GAChromosome.to_string(chromosome)
        if key in self._repo_thread:
            return
        if GAConstants.PARALLEL:
            t = threading.Thread(target=self._worker_fitness, args=(chromosome,))
            self._repo_thread[key] = t
        else:
            self._repo_thread[key] = None
            self._worker_fitness(chromosome)

    def _dequeue_all(self):
        if GAConstants.PARALLEL:
            for t in self._repo_thread.values():
                if t:
                    t.start()
            for t in self._repo_thread.values():
                if t:
                    t.join()
        chromosomes = []
        fitnesses   = []
        for chromo, fit in self._repo_fitness.values():
            chromosomes.append(chromo)
            fitnesses.append(fit)
        self._repo_thread.clear()
        self._repo_fitness.clear()
        return chromosomes, fitnesses

    # ------------------------------------------------------------------
    # GA main loop
    # ------------------------------------------------------------------
    def run(self) -> tuple:
        C = GAConstants

        self._log("Initializing population...")
        for _ in range(C.GA_POOL_SIZE):
            self._enqueue(GAChromosome(self.n_images))
        pool, fitness = self._dequeue_all()
        pool, fitness = self._sort(pool, fitness)

        eden_pool, eden_fitness = [], []
        timing = []

        for epoch in range(C.GA_EPOCHS):
            if self._stop:
                break

            start = time.time()

            for c in range(min(len(pool), C.GA_POOL_SIZE)):
                r = random()
                if r < C.GA_P_CROSSOVER and c > 0:
                    o1, o2 = GAChromosome.crossover(pool[c], pool[randrange(c)])
                    if o1.is_valid(): self._enqueue(o1)
                    if o2.is_valid(): self._enqueue(o2)

                if random() < C.GA_P_MUTATION:
                    off = GAChromosome.mutate(pool[c])
                    if off.is_valid(): self._enqueue(off)

            new_c, new_f = self._dequeue_all()
            pool    += new_c
            fitness += new_f
            pool, fitness = self._sort(pool, fitness)
            pool    = pool[:C.GA_POOL_SIZE]
            fitness = fitness[:C.GA_POOL_SIZE]

            eden_pool.append(GAChromosome.clone(pool[0]))
            eden_fitness.append(fitness[0])

            minimum, maximum = fitness[0], fitness[-1]
            to_reset = int(C.GA_P_CATACLYSM * len(pool))
            if abs(maximum - minimum) < C.GA_T_CATACLYSM:
                self._log(f"Epoch {epoch+1}: Type 1 Cataclysm!")
                for _ in range(to_reset):
                    self._enqueue(GAChromosome(self.n_images))
                new_c, new_f = self._dequeue_all()
                pool[:to_reset]    = new_c
                fitness[:to_reset] = new_f

            if fitness[to_reset] >= C.INF:
                self._log(f"Epoch {epoch+1}: Type 2 Cataclysm!")
                for _ in range(len(pool) - to_reset):
                    self._enqueue(GAChromosome(self.n_images))
                new_c, new_f = self._dequeue_all()
                pool[to_reset:]    = new_c
                fitness[to_reset:] = new_f

            elapsed = time.time() - start
            timing.append(elapsed)
            eta = (C.GA_EPOCHS - (epoch + 1)) * np.mean(timing) / 60.0
            self._log(
                f"Epoch {epoch+1}/{C.GA_EPOCHS} | Best RMS: {fitness[0]:.5f} | "
                f"Active imgs: {eden_pool[-1].genes.count(True)} | ETA: {eta:.1f} min",
                epoch=epoch+1, best_rms=fitness[0]
            )

        eden_pool, eden_fitness = self._sort(eden_pool, eden_fitness)
        best     = eden_pool[0]
        best_fit = eden_fitness[0]

        K, dist = self._final_calibrate(best)
        self._save_best_xml(best, best_fit, K, dist)

        return best, best_fit, K, dist

    def _final_calibrate(self, chromosome: GAChromosome):
        indices = chromosome.active_indices()
        obj_pts = [self.obj_points[i] for i in indices]
        img_pts = [self.matched_points[i] for i in indices]
        flags = chromosome.get_flags()
        if flags & cv2.CALIB_USE_INTRINSIC_GUESS:
            f_guess = max(self.img_width, self.img_height)
            K_init = np.array([[f_guess, 0, self.img_width / 2.0],
                            [0, f_guess, self.img_height / 2.0],
                            [0, 0, 1.0]], dtype=np.float64)
            dist_init = np.zeros((5, 1), dtype=np.float64)
        else:
            K_init = None
            dist_init = None

        _, K, dist, _, _ = cv2.calibrateCamera(
            obj_pts, img_pts,
            (self.img_width, self.img_height),
            K_init, dist_init,
            flags=flags
        )
        return K, dist

    def _save_best_xml(self, chromosome, fitness, K, dist):
        new_K, _ = cv2.getOptimalNewCameraMatrix(
            K, dist, (self.img_width, self.img_height), 1,
            (self.img_width, self.img_height)
        )
        path = os.path.join(self.out_dir, "ga_best_calibration.xml")
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        fs.write("cameraMatrix", K)
        fs.write("newCameraMatrix", new_K)
        fs.write("distCoeffs", dist)
        fs.write("ERR_REPROJ", fitness)
        fs.write("N_IMAGES", chromosome.genes.count(True))
        fs.write("ACTIVE_INDICES", np.array(chromosome.active_indices(), dtype=int))
        fs.write("FLAGS", chromosome.get_flags())
        fs.release()
        self._log(f"✅ Best calibration saved to {path}")

    def stop(self):
        self._stop = True

    @staticmethod
    def _sort(pool, fitness):
        paired  = sorted(zip(fitness, pool), key=lambda x: x[0])
        fitness = [p[0] for p in paired]
        pool    = [p[1] for p in paired]
        return pool, fitness

    def _log(self, msg, epoch=None, best_rms=None):
        if self.progress_callback:
            self.progress_callback(epoch, best_rms, msg)
