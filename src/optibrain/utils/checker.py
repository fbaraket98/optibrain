from sklearn.utils.validation import check_X_y

from palma import Project
class ProjectPlanChecker(object):
    """
    ProjectPlanChecker is an object that checks the project plan.

    At the :meth:`~palma.project.Project.build` moment, this object \
    run several checks in order to see if the project plan is well designed.

    Here is an overview of the checks performed by the object:
        - :meth:`~palma.utils.checker.ProjectPlanChecker._check_arrays`\
        : see whether X and y attribute are compliant with \
        sklearn standards.
        - :meth:`~palma.utils.checker.ProjectPlanChecker._check_project_problem`: see if the problem type is correctly \
        informed by the user.
        - :meth:`~palma.utils.checker.ProjectPlanChecker._check_problem_metrics`: see if the known metrics are consistent with \
        the project problem
    """

    def _check_arrays(self, project: Project) -> None:
        _, _ = check_X_y(project.X,
                         project.y,
                         dtype=None,
                         force_all_finite='allow-nan', multi_output=True)

    def _check_project_problem(self, project: Project) -> None:
        if not project.problem in ["classification", "regression"]:
            raise ValueError(
                f"Unknown problem: {project.problem}, please see documentation"
            )

    def run_checks(self, project: Project) -> None:
        """
        Perform some tests on the project plan

        Several checks are performed in order to check if the
        project plan is consistent:
            - checks the project problem
            - checks the metrics provided by the user
            - checks the data provided by the user (scikit learn wrapper)

        Parameters
        ----------
        project : :class:`~autolm.project.Project`
            an Project instance
        """
        self._check_project_problem(project)
        self._check_arrays(project)