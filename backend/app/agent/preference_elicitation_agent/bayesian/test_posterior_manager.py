"""
Unit tests for PosteriorManager and PosteriorDistribution.

Tests are schema-agnostic: fixtures use a small 3-dimension setup so the
same test logic works regardless of how many dimensions the live schema has.
"""

import pytest
import numpy as np
from app.agent.preference_elicitation_agent.bayesian.posterior_manager import (
    PosteriorDistribution,
    PosteriorManager
)

# --- shared test dimensions (deliberately NOT the 7 old ones) ---
TEST_DIMS = ["financial_importance", "work_environment_importance", "career_growth_importance"]
N = len(TEST_DIMS)


@pytest.fixture
def prior_mean():
    return np.array([0.5] * N)


@pytest.fixture
def prior_cov():
    return np.eye(N) * 0.5


@pytest.fixture
def posterior_dist():
    return PosteriorDistribution(
        mean=[0.5] * N,
        covariance=[[0.5 if i == j else 0.0 for j in range(N)] for i in range(N)],
        dimensions=TEST_DIMS,
    )


@pytest.fixture
def manager(prior_mean, prior_cov):
    return PosteriorManager(prior_mean=prior_mean, prior_cov=prior_cov, dimensions=TEST_DIMS)


class TestPosteriorDistribution:

    def test_init(self):
        dims = ["dim_a", "dim_b"]
        posterior = PosteriorDistribution(
            mean=[0.5, 0.5],
            covariance=[[1.0, 0.0], [0.0, 1.0]],
            dimensions=dims,
        )
        assert len(posterior.mean) == 2
        assert len(posterior.covariance) == 2
        assert posterior.dimensions == dims

    def test_dimensions_are_not_hardcoded(self, posterior_dist):
        """Dimensions must come from constructor, not a class-level default."""
        assert posterior_dist.dimensions == TEST_DIMS

    def test_get_variance(self, posterior_dist):
        variance = posterior_dist.get_variance("financial_importance")
        assert variance == 0.5

    def test_get_variance_all_dimensions(self, posterior_dist):
        for dim in posterior_dist.dimensions:
            assert posterior_dist.get_variance(dim) >= 0

    def test_get_correlation_diagonal(self, posterior_dist):
        corr = posterior_dist.get_correlation(TEST_DIMS[0], TEST_DIMS[0])
        assert np.isclose(corr, 1.0)

    def test_get_correlation_off_diagonal(self, posterior_dist):
        corr = posterior_dist.get_correlation(TEST_DIMS[0], TEST_DIMS[1])
        assert np.isclose(corr, 0.0)  # diagonal covariance → zero correlation

    def test_get_correlation_with_zero_variance(self):
        cov = [[0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0],
               [0.0, 0.0, 1.0]]
        posterior = PosteriorDistribution(mean=[0.5] * 3, covariance=cov, dimensions=TEST_DIMS)
        corr = posterior.get_correlation(TEST_DIMS[0], TEST_DIMS[1])
        assert corr == 0.0

    def test_sample_shape(self, posterior_dist):
        samples = posterior_dist.sample(n_samples=10)
        assert samples.shape == (10, N)

    def test_sample_statistics(self, posterior_dist):
        np.random.seed(42)
        samples = posterior_dist.sample(n_samples=10_000)
        sample_mean = np.mean(samples, axis=0)
        assert np.allclose(sample_mean, np.array(posterior_dist.mean), atol=0.05)

    def test_any_number_of_dimensions(self):
        """PosteriorDistribution must work for arbitrary N."""
        for n in [2, 4, 6]:
            dims = [f"dim_{i}" for i in range(n)]
            pd = PosteriorDistribution(
                mean=[0.5] * n,
                covariance=np.eye(n).tolist(),
                dimensions=dims,
            )
            assert len(pd.dimensions) == n
            assert pd.sample(n_samples=5).shape == (5, n)


class TestPosteriorManager:

    def test_init(self, manager, prior_mean):
        assert manager.posterior is not None
        assert len(manager.posterior.mean) == N
        assert np.allclose(manager.posterior.mean, prior_mean.tolist())
        assert manager.posterior.dimensions == TEST_DIMS

    def test_dimensions_mismatch_raises(self, prior_mean, prior_cov):
        with pytest.raises(ValueError, match="prior_mean length"):
            PosteriorManager(
                prior_mean=prior_mean,
                prior_cov=prior_cov,
                dimensions=["only_one_dim"],
            )

    def test_update_returns_posterior(self, manager):
        def simple_likelihood(obs, beta):
            return 0.5

        posterior = manager.update(likelihood_fn=simple_likelihood, observation={})
        assert isinstance(posterior, PosteriorDistribution)
        assert posterior.dimensions == TEST_DIMS

    def test_update_preserves_dimensions(self, manager):
        def simple_likelihood(obs, beta):
            return 0.7

        updated = manager.update(likelihood_fn=simple_likelihood, observation={})
        assert updated.dimensions == TEST_DIMS

    def test_update_changes_posterior(self, manager, prior_mean):
        def informative_likelihood(obs, beta):
            return 1.0 / (1.0 + np.exp(-beta[0]))

        original_mean = list(manager.posterior.mean)
        manager.update(likelihood_fn=informative_likelihood, observation={"choice": "high"})
        assert not np.allclose(manager.posterior.mean, original_mean)

    def test_numerical_gradient_shape(self, manager):
        def test_likelihood(obs, beta):
            return np.exp(-np.sum(beta ** 2))

        beta = np.array([0.5] * N)
        grad = manager._numerical_gradient(test_likelihood, {}, beta)
        assert grad.shape == (N,)
        assert np.all(np.isfinite(grad))

    def test_numerical_hessian_symmetry(self, manager):
        def test_likelihood(obs, beta):
            return np.exp(-np.sum(beta ** 2))

        beta = np.array([0.5] * N)
        hess = manager._numerical_hessian(test_likelihood, {}, beta)
        assert np.allclose(hess, hess.T)

    def test_find_map_convergence(self, manager, prior_mean, prior_cov):
        def quadratic_likelihood(obs, beta):
            return np.exp(-np.sum((beta - 1.0) ** 2))

        map_estimate = manager._find_map(
            likelihood_fn=quadratic_likelihood,
            observation={},
            prior_mean=prior_mean,
            prior_cov=prior_cov,
        )
        assert np.all(map_estimate > 0.4)
        assert np.all(map_estimate < 1.1)

    def test_update_with_multiple_observations(self, manager):
        def consistent_likelihood(obs, beta):
            return 1.0 / (1.0 + np.exp(-(beta[0] - 1.0)))

        for _ in range(3):
            manager.update(likelihood_fn=consistent_likelihood, observation={})

        assert manager.posterior.mean[0] > 0.5

    def test_update_reduces_uncertainty(self, manager):
        def informative_likelihood(obs, beta):
            return np.exp(-10 * np.sum((beta - 0.7) ** 2))

        initial_variance = manager.posterior.get_variance(TEST_DIMS[0])
        manager.update(likelihood_fn=informative_likelihood, observation={})
        updated_variance = manager.posterior.get_variance(TEST_DIMS[0])
        assert updated_variance < initial_variance

    def test_singular_hessian_handling(self, manager):
        def constant_likelihood(obs, beta):
            return 0.5

        posterior = manager.update(likelihood_fn=constant_likelihood, observation={})
        assert isinstance(posterior, PosteriorDistribution)

    def test_numerical_stability_extreme_values(self, manager):
        def extreme_likelihood(obs, beta):
            return 1.0 / (1.0 + np.exp(-np.sum(beta)))

        manager.posterior.mean = [10.0] * N

        posterior = manager.update(likelihood_fn=extreme_likelihood, observation={})
        assert all(np.isfinite(posterior.mean))
        assert all(all(np.isfinite(row)) for row in posterior.covariance)

    def test_posterior_covariance_positive_definite(self, manager):
        def normal_likelihood(obs, beta):
            return np.exp(-np.sum(beta ** 2))

        manager.update(likelihood_fn=normal_likelihood, observation={})
        cov_matrix = np.array(manager.posterior.covariance)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        assert np.all(eigenvalues > -1e-6)

    def test_laplace_approximation_symmetry(self, manager):
        def simple_likelihood(obs, beta):
            return 0.7

        manager.update(likelihood_fn=simple_likelihood, observation={})
        cov_matrix = np.array(manager.posterior.covariance)
        assert np.allclose(cov_matrix, cov_matrix.T)

    def test_prior_influence_weakens_with_data(self, prior_mean, prior_cov):
        manager1 = PosteriorManager(prior_mean=prior_mean, prior_cov=prior_cov * 0.1, dimensions=TEST_DIMS)
        manager2 = PosteriorManager(prior_mean=prior_mean, prior_cov=prior_cov * 0.1, dimensions=TEST_DIMS)

        def data_likelihood(obs, beta):
            return np.exp(-100 * (beta[0] - 0.9) ** 2)

        manager1.update(likelihood_fn=data_likelihood, observation={})
        for _ in range(5):
            manager2.update(likelihood_fn=data_likelihood, observation={})

        assert abs(manager2.posterior.mean[0] - 0.9) < abs(manager1.posterior.mean[0] - 0.9)
